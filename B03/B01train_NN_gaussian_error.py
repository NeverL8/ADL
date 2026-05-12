from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import typing as tp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchsummary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_examples import UncertaintyCNN
from helper import denormalize, denormalize_std, train_model, get_normalized_data, evaluate_model

dblArr: tp.TypeAlias = np.typing.NDArray[np.float64]

DATA_PATH = "../data/4/"
model_name = "CNN_1_gaussian_error"

# Hyperparameters
learning_rate = 0.8e-5
batch_size = 32
num_epochs = 400
patience = 20  # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7  # Fraction of the data used for training
val_fraction = 0.15  # Fraction of the data used for validation


# tensor object and dataloader creation (analogue to last exercise)
def createDataLoaders(
    spectra: dblArr, labels: dblArr, batch_size: int = batch_size, train_fraction: float = train_fraction, val_fraction: float = val_fraction
) -> tp.Tuple[DataLoader, DataLoader, DataLoader]:
    # Convert numpy arrays to PyTorch tensors
    spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Split the data into training, validation, and test sets
    total_samples = len(spectra_tensor)
    train_size = int(train_fraction * total_samples)
    val_size = int(val_fraction * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        TensorDataset(spectra_tensor, labels_tensor), [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders for batching and shuffling the data
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


# Define the negative log-likelihood (NLL) loss function.
# This loss function is appropriate for regression tasks where we predict both values and uncertainties.
def nll_loss(inputs, batch_labels, model, n_labels: int) -> torch.Tensor:
    """
    Calculate the negative log-likelihood (NLL) loss.

    Parameters
    ----------
    inputs : torch.Tensor
        The input tensor to the model.
    batch_labels : torch.Tensor
        The ground truth labels for the batch.
    model : nn.Module
        The neural network model.

    Returns
    -------
    torch.Tensor
        The calculated NLL loss.
    """

    predictions = model(inputs)

    mean = predictions[:, :n_labels]  # Extract the mean values
    log_std = predictions[:, n_labels:]  # Extract the log standard deviations
    std = torch.exp(log_std)  # Convert log standard deviation to standard deviation

    return torch.mean((0.5 * ((batch_labels - mean) / std) ** 2) + log_std)  # NLL formula


def predictionEvaluation(n_labels: int, ranges: dblArr, evaluation_results: tp.Tuple[dblArr, dblArr, None, None]) -> tp.Tuple[dblArr, dblArr, dblArr]:
    predictions, true_labels, _, _ = evaluation_results

    # Denormalize predictions, extract the predicted standard deviations & denormalize true labels
    return (
        denormalize(predictions[:, :n_labels], ranges),  # Denormalize predictions
        denormalize_std(np.exp(predictions[:, n_labels:]), ranges),  # Extract the predicted standard deviations
        denormalize(true_labels, ranges),  # Denormalize true labels
    )


# Function to plot training and validation losses, scatter plots, pull distributions, and true vs predicted distributions
def evaluationPlotting(
    train_losses: list[float], val_losses: list[float], all_true_labels: dblArr, pred_mean: dblArr, pred_std: dblArr, labelNames: list[str], n_labels: int
) -> None:
    # Check if the "plots" directory exists, if not, create it
    if not os.path.exists("plots/%s" % model_name):
        os.makedirs("plots/%s" % model_name)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    # plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/training_validation_loss.png")

    # Scatter plots for predictions
    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels
        plt.scatter(gt[:, j], y=pred_mean[:, j], s=6, alpha=0.2)
        plt.plot(
            [gt[:, j].min().item(), gt[:, j].max().item()],
            [gt[:, j].min().item(), gt[:, j].max().item()],
            c="black",
            linestyle="dashed",
            label="Perfect prediction",
        )
        plt.xlabel("true " + labelNames[j])
        plt.ylabel("predicted " + labelNames[j])
        plt.legend()
    plt.tight_layout()
    plt.savefig("plots/scatter.png")

    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        diff = gt - pred
        plt.hist(diff, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xlabel(f"True - Predicted {labelNames[j]}")
        plt.ylabel("Frequency")
        plt.title(f"Distribution for {labelNames[j]}")
        plt.axvline(diff.mean(), color="red", linestyle="dashed", linewidth=1)
        plt.axvline(diff.std(), color="green", linestyle="dashed", linewidth=1)
        plt.text(
            0.95,
            0.95,
            f"Mean: {diff.mean():.2f}\nStd: {diff.std():.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    plt.tight_layout()
    plt.savefig("plots/true_predicted.png")

    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        plt.hist(std, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xlabel(f"STD for {labelNames[j]}")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of STD for {labelNames[j]}")
        plt.axvline(std.mean(), color="red", linestyle="dashed", linewidth=1)
        plt.axvline(std.std(), color="green", linestyle="dashed", linewidth=1)
        plt.text(
            0.95,
            0.95,
            f"Mean: {std.mean():.2f}\nStd: {std.std():.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    plt.tight_layout()
    plt.savefig("plots/std.png")

    # Plot pull distributions for the three labels
    plt.figure(figsize=(16, 7.5))
    for j in range(n_labels):
        plt.subplot(1, 3, j + 1)
        gt = all_true_labels[:, j]
        pred = pred_mean[:, j]
        std = pred_std[:, j]  # Extract the predicted standard deviations
        pull = (gt - pred) / std  # Calculate the pull
        plt.hist(pull, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.xlabel(f"Pull for {labelNames[j]}")
        plt.ylabel("Frequency")
        plt.title(f"Pull Distribution for {labelNames[j]}")
        plt.axvline(pull.mean(), color="red", linestyle="dashed", linewidth=1)
        plt.axvline(pull.std(), color="green", linestyle="dashed", linewidth=1)
        plt.text(
            0.95,
            0.95,
            f"Mean: {pull.mean():.2f}\nStd: {pull.std():.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    plt.tight_layout()
    plt.savefig("plots/pull.png")

    plt.show()
    return


def main() -> int:
    # Call the function to get normalized data
    spectra, labels, spectra_length, n_labels, labelNames, ranges = get_normalized_data(DATA_PATH)  # pyright: ignore[reportAssignmentType]

    model = UncertaintyCNN(n_labels * 2)  # times two because we want to predict the mean and the standard deviation

    # Detect and use Apple Silicon GPU (MPS) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Print the model summary before moving it to the device
    summary(model, input_size=(1, spectra_length))
    print(f"Using device: {device}")
    model.to(device)

    # Prepare data
    train_loader, val_loader, test_loader = createDataLoaders(spectra, labels)

    # Call the function
    loss_function = partial(nll_loss, n_labels=n_labels)
    train_losses, val_losses, best_model = train_model(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, patience, device)

    # Final evaluation on the test dataset
    model.load_state_dict(best_model)  # pyright: ignore[reportArgumentType] # Load the best model
    # Save the best model to the "models" directory
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(best_model, f"models/{model_name}_best.pth")
    model.to(device)

    # evaluate the model predictions
    pred_mean, pred_std, all_true_labels = predictionEvaluation(n_labels, ranges, evaluate_model(model, test_loader, loss_function, device))

    # present results in plots
    evaluationPlotting(train_losses, val_losses, all_true_labels, pred_mean, pred_std, labelNames, n_labels)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
