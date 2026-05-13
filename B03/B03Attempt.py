import argparse
import numpy as np
from matplotlib import pyplot as plt
import jammy_flows
from scipy.stats import norm
from helper import get_normalized_data
import torch
import sys,os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchsummary import summary
#from functools import partial
import typing as tp


from helper import denormalize, denormalize_std, train_model, get_normalized_data, evaluate_model

dblArr: tp.TypeAlias = np._typing.NDArray[np.float64]


DATA_PATH = "/home/simon/ADL/B03/"
fp64_on_cpu = True

# Hyperparameters
learning_rate = 1e-5
batch_size = 128
num_epochs = 20
patience = 10  # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7  # Fraction of the data used for training
val_fraction = 0.15  # Fraction of the data used for validation


# tensor object and dataloader creation (analogue to last exercise)
# def createDataLoaders(
#     spectra: dblArr, labels: dblArr, batch_size: int = batch_size, train_fraction: float = train_fraction, val_fraction: float = val_fraction
# ) -> tp.Tuple[DataLoader, DataLoader, DataLoader]:
#     # Convert numpy arrays to PyTorch tensors
#     spectra_tensor = spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
#     labels_tensor = torch.tensor(labels, dtype=torch.float32)

#     # Split the data into training, validation, and test sets
#     total_samples = len(spectra_tensor)
#     train_size = int(train_fraction * total_samples)
#     val_size = int(val_fraction * total_samples)
#     test_size = total_samples - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(
#         TensorDataset(spectra_tensor, labels_tensor), [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
#     )

#     # Create DataLoaders for batching and shuffling the data
#     return (
#         DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
#         DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
#         DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
#     )

def createDataLoaders(spectra, labels, batch_size=batch_size,
                      train_fraction=train_fraction,
                      val_fraction=val_fraction):

    spectra_tensor = torch.tensor(spectra, dtype=torch.float32)   # [N, L]
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    if spectra_tensor.ndim == 3:
        spectra_tensor = spectra_tensor.squeeze(1)

    dataset = TensorDataset(spectra_tensor, labels_tensor)

    total = len(dataset)
    train_size = int(train_fraction * total)
    val_size = int(val_fraction * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
    )


# Call the function to get normalized data


# Define the CNN encoder model. The output of the model is the input to the normalizing flow.
# The latent dimension is the number of parameters in the normalizing flow.
# class TinyCNNEncoder(nn.Module):
#     def __init__(self, latent_dimension):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm1d(16),
#             nn.MaxPool1d(2),

#             nn.Conv1d(16, 32, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm1d(32),
#             nn.MaxPool1d(2),

#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),

#             nn.AdaptiveAvgPool1d(16)
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 16, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, latent_dimension)
#         )

#     def forward(self, x):
#         print(x.shape)
#         x = self.conv(x)
#         print(x.shape)
#         x = self.fc(x)
#         print(x.shape)
#         return x

class TinyCNNEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.AdaptiveAvgPool1d(16)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dimension)
        )

    def forward(self, x):
        # ✅ SINGLE RESPONSIBILITY: enforce Conv1d format
        if x.dim() == 2:
            x = x.unsqueeze(1)        # [B, L] → [B, 1, L]
        elif x.dim() == 4:
            x = x.squeeze(1)          # fixes accidental [B,1,1,L]

        if x.dim() != 3:
            raise ValueError(f"Expected [B,1,L], got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Channel must be 1, got {x.shape}")

        x = self.conv(x)
        x = self.fc(x)
        return x

def nf_loss(inputs, batch_labels, model) -> torch.Tensor:
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(batch_labels, inputs)  # get the probability of the labels given the input data
    loss = -log_pdfs.mean()  # take the negative mean of the log probabilities
    return loss


# Defining the normalizing flow model is a bit more involved and requires knowledge of the jammy_flows library.
# Therefore, we provide the relevant code here.
class CombinedModel(nn.Module):
    """
    A combined model that integrates a normalizing flow with a CNN encoder.
    """

    def __init__(self, encoder, nf_type="full_flow"):
        """
        Initializes the normalizing flow model.

        Parameters
        ----------
        encoder : callable
            A function or callable object that returns an encoder model. The encoder model
            should take the number of flow parameters as input and output the latent dimension.
            I.e. the `TinyCNNEncoder` class defined above.
        nf_type : str, optional
            The type of normalizing flow to use. Options are "diagonal_gaussian", "full_gaussian",
            and "full_flow". Default is "diagonal_gaussian".
        Raises
        ------
        Exception
            If an unknown `nf_type` is provided.
        Notes
        -----
        This method sets up a 3-dimensional probability density function (PDF) over Euclidean space (e3)
        using the specified normalizing flow type. The flow structure and options are configured based on
        the provided `nf_type`. The PDF is created using the `jammy_flows` library, and the number of flow
        parameters is determined and printed. The encoder is initialized with the number of flow parameters.
        """

        super().__init__()

        # we define a 3-d PDF over Euclidean space (e3)
        # using recommended settings (https://github.com/thoglu/jammy_flows/issues/5 scroll down)
        opt_dict = {}
        opt_dict["t"] = {}
        if nf_type == "diagonal_gaussian":
            opt_dict["t"]["cov_type"] = "diagonal"
            flow_defs = "t"
        elif nf_type == "full_gaussian":
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "t"
        elif nf_type == "full_flow":
            opt_dict["t"]["cov_type"] = "full"
            flow_defs = "gggt"
        else:
            raise Exception("Unknown nf type ", nf_type)

        opt_dict["g"] = dict()
        opt_dict["g"]["fit_normalization"] = 1
        opt_dict["g"]["upper_bound_for_widths"] = 1.0
        opt_dict["g"]["lower_bound_for_widths"] = 0.01

        self.nf_type = nf_type

        # 3d PDF (e3) with ggggt flow structure. Four Gaussianization-flow (https://arxiv.org/abs/2003.01941) layers ("g") and an affine flow ("t")
        self.pdf = jammy_flows.pdf("e3", flow_defs, options_overwrite=opt_dict, amortize_everything=True, amortization_mlp_use_custom_mode=True)

        # get the number of flow parameters
        num_flow_parameters = self.pdf.total_number_amortizable_params

        print("The normalizing flow has ", num_flow_parameters, " parameters...")

        # latent dimension (output of the CNN encoder) is set to 128
        self.encoder = encoder(num_flow_parameters)

    def log_pdf_evaluation(self, target_labels, input_data):
        """
        Evaluate the log probability density function (PDF) for the given target labels and input data.

        The normalizing flow parameters are predicted by the encoder network based on the input data.
        Then, the log PDF is evaluated at the position of the label.

        Parameters:
        -----------
        target_labels : torch.Tensor
            The target labels for which the log PDF is to be evaluated.
        input_data : torch.Tensor
            The input data to be encoded and used for evaluating the log PDF.
        Returns:
        --------
        log_pdf : torch.Tensor
            The evaluated log PDF for the given target labels and input data.
        """
        latent_intermediate = self.encoder(input_data)  # get the flow parameters from the CNN encoder

        if self.nf_type == "full_flow":
            # convert to double. Double precision is needed for the Gaussianization flow. This is for numerical stability.
            if fp64_on_cpu:  # MPS does not support double precision, therefore we need to run the flow on the CPU
                latent_intermediate = latent_intermediate.cpu().to(torch.float64)
                target_labels = target_labels.cpu().to(torch.float64)
            else:
                latent_intermediate = latent_intermediate.to(torch.float64)
                target_labels = target_labels.to(torch.float64)

        # evaluate the log PDF at the target labels
        log_pdf, _, _ = self.pdf(target_labels, amortization_parameters=latent_intermediate)
        return log_pdf

    def sample(self, flow_params, samplesize_per_batchitem=1000):
        """
        Sample new points from the PDF given input data.

        Parameters
        ----------
        flow_params : tensor
            Parameters for the normalizing flow, must be of shape (B, L) where B is the batch size and L is the latent dimension.
        samplesize_per_batchitem : int, optional
            Number of samples to draw per batch item. Defaults to 1000.

        Returns
        -------
        tensor
            A tensor of shape (B, S, D) where B is the batch dimension, S is the number of samples,
            and D is the dimension of the target space for the samples.
        """
        # for full flow we need to convert to double precision for the normalizing flow
        # for numerical stability
        if self.nf_type == "full_flow":
            # convert to double
            if fp64_on_cpu:  # MPS does not support double precision, therefore we need to run the flow on the CPU
                flow_params = flow_params.cpu().to(torch.float64)
            else:
                flow_params = flow_params.to(torch.float64)

        batch_size = flow_params.shape[0]  # get the batch size
        # sample from the normalizing flow
        repeated_samples, _, _, _ = self.pdf.sample(
            amortization_parameters=flow_params.repeat_interleave(samplesize_per_batchitem, dim=0), allow_gradients=False
        )

        # reshape the samples to be grouped by batch item
        reshaped_samples = repeated_samples.view(batch_size, samplesize_per_batchitem, -1)#[:, None, :]

        return reshaped_samples

    def forward(self, input_data, samplesize_per_batchitem=1000):
        """
        Perform a forward pass through the model, predicting the mean and standard deviation of the samples.

        Normalizing flows do not directly predict the target labels. Instead, they predict the parameters of the flow that
        transforms the base distribution to the target distribution. Often, we still want to predict the target labels.
        Then, we can sample from the distribution and form the mean of the samples and their standard deviations.
        This is what this function does.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor.
        Returns
        -------
        torch.Tensor
            A tensor of size (B, D*2) where the first half (size D) are the means,
            the second half (another D) are the standard deviations.
        """
        flow_params = self.encoder(input_data)
        samples = self.sample(flow_params, samplesize_per_batchitem=samplesize_per_batchitem)

        # form mean along dim 1 (samples)
        means = samples.mean(dim=1)
        # form std along dim 1 (samples)
        std_deviations = samples.std(dim=1)

        # return means and std deviations as a concatenated tensor along dim 1
        return torch.cat([means, std_deviations], dim=1)

    def visualize_pdf(self, input_data, filename, samplesize=1000, batch_index=0, truth=None):
        """
        Visualizes the probability density function (PDF) of the given input data using a normalizing flow model.

        The function generates samples from the normalizing flow (using the sample() function)
        and plots the histogram of the samples together with a Gaussian approximation.

        Parameters
        ----------
        input_data : torch.Tensor
            The input data tensor from which to pick one batch item for visualization.
        filename : str
            The filename where the resulting plot will be saved.
        samplesize : int, optional
            The number of samples to generate for the PDF visualization (default is 10000).
        batch_index : int, optional
            The index of the batch item to visualize (default is 0).
        truth : torch.Tensor, optional
            The true values of the labels, used for comparison in the plot (default is None).

        Returns
        -------
        None
        """
        # pick out one input from batch
        input_bitem = input_data[batch_index : batch_index + 1]
        if input_bitem.dim() == 1:
            input_bitem = input_bitem.unsqueeze(0)

        # get the flow parameters (by passing the input data through the CNN encoder network)
        flow_params = self.encoder(input_bitem)

        # sample from the normalizing flow (i.e. samples are drawn from the base distribution and transformed by the flow
        # using the change-of-variable formula)
        samples = self.sample(flow_params, samplesize_per_batchitem=samplesize)
        # the rest of the code is just plotting.

        # we only have 1 batch item
        samples = samples.squeeze(0)

        # plot three 1-dimensional distributions together with normal approximation,
        # so we calculate the mean and std of the samples
        mean = samples.mean(dim=0).cpu().numpy()
        std = samples.std(dim=0).cpu().numpy()
        samples = samples.cpu().numpy()

        fig, axdict = plt.subplots(3, 1)
        for dim_ind in range(3):
            # plot the histogram of the samples
            axdict[dim_ind].hist(samples[:, dim_ind], color="k", density=True, bins=50, alpha=0.5, label="density based on samples")

            # plot the Gaussian approximation
            min_sample = samples[:, dim_ind].min()
            max_sample = samples[:, dim_ind].max()
            xvals = np.linspace(min_sample, max_sample, 1000)
            yvals = norm.pdf(xvals, loc=mean[dim_ind], scale=std[dim_ind])
            axdict[dim_ind].plot(xvals, yvals, color="green", label="Gaussian approximation")

            # plot the true value if it is given
            if truth is not None:
                true_value = truth[dim_ind].cpu().item()
                axdict[dim_ind].axvline(true_value, color="red", label="true value")

            # plot the legend only for the first panel
            if dim_ind == 0:
                axdict[dim_ind].legend()

        plt.savefig(filename)
        plt.close(fig)

def predictionEvaluation(n_labels: int, ranges: dblArr, evaluation_results: tp.Tuple[dblArr, dblArr, None, None]) -> tp.Tuple[dblArr, dblArr, dblArr]:
    predictions, true_labels, _, _ = evaluation_results

    # Denormalize predictions, extract the predicted standard deviations & denormalize true labels
    return (
        denormalize(predictions[:, :n_labels], ranges),  # Denormalize predictions
        denormalize_std(np.exp(predictions[:, n_labels:]), ranges),  # Extract the predicted standard deviations
        denormalize(true_labels, ranges),  # Denormalize true labels
    )

# def predictionEvaluation(n_labels: int, ranges: dblArr, evaluation_results: tp.Tuple[dblArr, dblArr, None, None]) -> tp.Tuple[dblArr, dblArr, dblArr]:
#     predictions, true_labels, _, _ = evaluation_results

#     # Denormalize predictions, extract the predicted standard deviations & denormalize true labels
#     return (
#         denormalize(predictions[:, :n_labels], ranges),  # Denormalize predictions
#         denormalize_std(np.exp(predictions[:, n_labels:]), ranges),  # Extract the predicted standard deviations
#         denormalize(true_labels, ranges),  # Denormalize true labels
#     )


# # Function to plot training and validation losses, scatter plots, pull distributions, and true vs predicted distributions
# def evaluationPlotting(
#     train_losses: list[float], val_losses: list[float], all_true_labels: dblArr, pred_mean: dblArr, pred_std: dblArr, labelNames: list[str], n_labels: int
# ) -> None:
#     # Check if the "plots" directory exists, if not, create it
#     if not os.path.exists("plots/%s" % model_name):
#         os.makedirs("plots/%s" % model_name)

#     # Plot training and validation loss
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
#     # plt.yscale('log')
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("plots/training_validation_loss.png")

#     # Scatter plots for predictions
#     plt.figure(figsize=(16, 7.5))
#     for j in range(n_labels):
#         plt.subplot(1, 3, j + 1)
#         gt = all_true_labels
#         plt.scatter(gt[:, j], y=pred_mean[:, j], s=6, alpha=0.2)
#         plt.plot(
#             [gt[:, j].min().item(), gt[:, j].max().item()],
#             [gt[:, j].min().item(), gt[:, j].max().item()],
#             c="black",
#             linestyle="dashed",
#             label="Perfect prediction",
#         )
#         plt.xlabel("true " + labelNames[j])
#         plt.ylabel("predicted " + labelNames[j])
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig("plots/scatter.png")

#     plt.figure(figsize=(16, 7.5))
#     for j in range(n_labels):
#         plt.subplot(1, 3, j + 1)
#         gt = all_true_labels[:, j]
#         pred = pred_mean[:, j]
#         std = pred_std[:, j]  # Extract the predicted standard deviations
#         diff = gt - pred
#         plt.hist(diff, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
#         plt.xlabel(f"True - Predicted {labelNames[j]}")
#         plt.ylabel("Frequency")
#         plt.title(f"Distribution for {labelNames[j]}")
#         plt.axvline(diff.mean(), color="red", linestyle="dashed", linewidth=1)
#         plt.axvline(diff.std(), color="green", linestyle="dashed", linewidth=1)
#         plt.text(
#             0.95,
#             0.95,
#             f"Mean: {diff.mean():.2f}\nStd: {diff.std():.2f}",
#             transform=plt.gca().transAxes,
#             fontsize=12,
#             verticalalignment="top",
#             horizontalalignment="right",
#             bbox=dict(facecolor="white", alpha=0.5),
#         )
#     plt.tight_layout()
#     plt.savefig("plots/true_predicted.png")

#     plt.figure(figsize=(16, 7.5))
#     for j in range(n_labels):
#         plt.subplot(1, 3, j + 1)
#         gt = all_true_labels[:, j]
#         pred = pred_mean[:, j]
#         std = pred_std[:, j]  # Extract the predicted standard deviations
#         plt.hist(std, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
#         plt.xlabel(f"STD for {labelNames[j]}")
#         plt.ylabel("Frequency")
#         plt.title(f"Distribution of STD for {labelNames[j]}")
#         plt.axvline(std.mean(), color="red", linestyle="dashed", linewidth=1)
#         plt.axvline(std.std(), color="green", linestyle="dashed", linewidth=1)
#         plt.text(
#             0.95,
#             0.95,
#             f"Mean: {std.mean():.2f}\nStd: {std.std():.2f}",
#             transform=plt.gca().transAxes,
#             fontsize=12,
#             verticalalignment="top",
#             horizontalalignment="right",
#             bbox=dict(facecolor="white", alpha=0.5),
#         )
#     plt.tight_layout()
#     plt.savefig("plots/std.png")

#     # Plot pull distributions for the three labels
#     plt.figure(figsize=(16, 7.5))
#     for j in range(n_labels):
#         plt.subplot(1, 3, j + 1)
#         gt = all_true_labels[:, j]
#         pred = pred_mean[:, j]
#         std = pred_std[:, j]  # Extract the predicted standard deviations
#         pull = (gt - pred) / std  # Calculate the pull
#         plt.hist(pull, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
#         plt.xlabel(f"Pull for {labelNames[j]}")
#         plt.ylabel("Frequency")
#         plt.title(f"Pull Distribution for {labelNames[j]}")
#         plt.axvline(pull.mean(), color="red", linestyle="dashed", linewidth=1)
#         plt.axvline(pull.std(), color="green", linestyle="dashed", linewidth=1)
#         plt.text(
#             0.95,
#             0.95,
#             f"Mean: {pull.mean():.2f}\nStd: {pull.std():.2f}",
#             transform=plt.gca().transAxes,
#             fontsize=12,
#             verticalalignment="top",
#             horizontalalignment="right",
#             bbox=dict(facecolor="white", alpha=0.5),
#         )
#     plt.tight_layout()
#     plt.savefig("plots/pull.png")

#     plt.show()
#     return

def main() -> int:
    spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)  # pyright: ignore[reportAssignmentType]

    parser = argparse.ArgumentParser()
    parser.add_argument("-normalizing_flow_type", default="full_flow", choices=["diagonal_gaussian", "full_gaussian", "full_flow"])
    args = parser.parse_args()
    print("Using normalizing flow type ", args.normalizing_flow_type)

    model = CombinedModel(TinyCNNEncoder, nf_type=args.normalizing_flow_type)

    # Detect and use Apple Silicon GPU (MPS) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #if args.normalizing_flow_type == "full_flow" and device.type == "mps":
        # MPS does not support double precision, therefore we need to run the flow on the CPU
    #    fp64_on_cpu = True
    #summary(model.encoder, input_size=(1, spectra_length))
    print(f"Using device: {device}, performing fp64 on CPU: {fp64_on_cpu}")
    model.to(device)

    # Prepare data
    train_loader, val_loader, test_loader = createDataLoaders(spectra, labels)

    # Call the function
    loss_function = nf_loss
    train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience, device)

    # Final evaluation on the test dataset
    model.load_state_dict(best_model)  # pyright: ignore[reportArgumentType] # Load the best model
    # Save the best model to the "models" directory
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(best_model, f"models/{args.normalizing_flow_type}_best.pth")
    model.to(device)

#     # evaluate the model predictions
#     evaluation_results = evaluate_model(
#     model,
#     test_loader,
#     loss_function,
#     device
# )
    pred_mean, pred_std, all_true_labels = predictionEvaluation(n_labels, ranges, evaluate_model(model, test_loader, loss_function, device))
    z = (all_true_labels - pred_mean) / (pred_std)
    print("Z mean:", z.mean(axis=0))
    print("Z std:", z.std(axis=0))
   
    # present results in plots

    sample_spectrum = torch.tensor(
    spectra[:1],
    dtype=torch.float32).to(device)

    sample_truth = torch.tensor(
    labels[0],
    dtype=torch.float32).to(device)
    model.visualize_pdf(
    sample_spectrum,
    "idk.png",
    truth=sample_truth
    )
    #model.visualize_pdf(train_losses, val_losses, all_true_labels, pred_mean, pred_std, labelNames, n_labels)

    return os.EX_OK

if __name__ == "__main__":
    sys.exit(main())



