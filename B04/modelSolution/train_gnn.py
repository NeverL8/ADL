import sys, os
import argparse
import awkward
from datetime import datetime
import h5py as h5
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, NamedTuple, Callable

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

from gnn_encoder import GNNEncoder, collate_fn_gnn
from gnn_trafo_helper import train_model, evaluate_model, get_img_from_matplotlib, normalize_quantity, denormalize_quantity

# Hyperparameters
learning_rate = 0.8e-5
batch_size = 32
num_epochs = 1000
patience = 50  # Training loop with early stopping, if the validation loss does not improve for 'patience' epochs
train_fraction = 0.7  # Fraction of the data used for training
val_fraction = 0.15  # Fraction of the data used for validation

# define some global variables and types
DATA_PATH = "../IceCube2DDataset/"  # path to the data
model_name = "encoder_GNN"
data = NamedTuple("datasets", [("train", awkward.Array), ("val", awkward.Array), ("test", awkward.Array)])


# to get familiar with the dataset, let's inspect it.
def inspect_datasets(datasets: data) -> None:
    print(
        f"The training dataset contains {len(datasets.train)} events.\n"
        + f"The validation dataset contains {len(datasets.val)} events.\n"
        + f"The test dataset contains {len(datasets.test)} events.\n"
        + f"The training dataset has the following columns: {datasets.train.fields}\n"
        + f"The validation dataset has the following columns: {datasets.val.fields}\n"
        + f"The test dataset has the following columns: {datasets.test.fields}\n"
    )
    # print the first event of the training dataset
    print(f"The first event of the training dataset is: {datasets.train[0]}")

    # We are interested in the labels xpos and ypos. This is the position of the neutrino interaction that we want to predict.
    print(f"The first event of the training dataset has the following labels: {datasets.train['xpos'][0]}, {datasets.train['ypos'][0]}")
    # Awkward arrays also allow us to obtain the 'xpos' and 'ypos' label for all events in the dataset
    print(f"The first 10 labels of the training dataset are: {datasets.train['xpos'][:10]}, {datasets.train['ypos'][:10]}")

    # The data can be accessed by using the 'data' key.
    # The data is a 3D array with the first dimension being the number of events,
    # the second dimension being the the three features (time, x, y)
    # the third dimension being the number of hits,
    print(f"The first event of the training dataset has {len(datasets.train['data'][0][0])} hits, i.e., detected photons.")
    # Let's loop over all hits and print the time, x, and y coordinates of the first event.
    for i in range(len(datasets.train["data"][0, 0])):
        print(f"Hit {i}: time = {datasets.train['data'][0,0,i]}, x = {datasets.train['data'][0,1, i]}, y = {datasets.train['data'][0,2,i]}")
    # To get all hit times of the first event, you can use the following code:
    print(
        f"The first event of the training dataset has the following hit times: {datasets.train['data'][0, 0]}\n"
        + f"The first event of the training dataset has the following hit x positions: {datasets.train['data'][0, 1]}\n"
        + f"The first event of the training dataset has the following hit y positions: {datasets.train['data'][0, 2]}\n"
    )

    return


# Normalize Awkward data
def normalize_dataset(dataset: awkward.Array, normalization_file: str) -> awkward.Array:
    # working with Awkward arrays is a bit tricky because the ['data'] field can't be assigned in-place,
    # so we need to extract the time, x, and y coordinates, normalize them separately,
    # and then concatenate them back together.

    # Normalize time, x and y coordinates separately
    if not os.path.isfile(normalization_file):
        norm_times, time_norm_params = normalize_quantity(dataset["data"][:, 0:1, :])
        norm_x, x_norm_params = normalize_quantity(dataset["data"][:, 1:2, :])
        norm_y, y_norm_params = normalize_quantity(dataset["data"][:, 2:3, :])

        with h5.File(normalization_file, "w") as file:
            for quantity in zip(["time", "x", "y"], [time_norm_params, x_norm_params, y_norm_params]):
                group = file.create_group(quantity[0])
                group.create_dataset("mean", data=quantity[1][0])
                group.create_dataset("standard", data=quantity[1][1])

    else:
        with h5.File(normalization_file, "r") as file:
            norm_params = dict(
                [
                    (group.name.split("/")[-1], (np.array(group.get("mean")), np.array(group.get("standard"))))  # pyright: ignore[reportOptionalMemberAccess]
                    for key in file.keys()
                    if isinstance((group := file.get(key)), h5.Group)
                ]
            )
        print(norm_params)

        norm_times, time_norm_params = normalize_quantity(dataset["data"][:, 0:1, :], norm_params["time"])  # pyright: ignore[reportArgumentType]
        norm_x, x_norm_params = normalize_quantity(dataset["data"][:, 1:2, :], norm_params["x"])  # pyright: ignore[reportArgumentType]
        norm_y, y_norm_params = normalize_quantity(dataset["data"][:, 2:3, :], norm_params["y"])  # pyright: ignore[reportArgumentType]

    # Concatenate the normalized data back together
    dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)

    # Normalize labels (this can be done in-place)
    dataset["xpos"] = (dataset["xpos"] - x_norm_params[0]) / x_norm_params[1]
    dataset["ypos"] = (dataset["ypos"] - y_norm_params[0]) / y_norm_params[1]

    return dataset


def denormalize_dataset(quantity: np.ndarray, normalization_file: str) -> np.ndarray:
    with h5.File(normalization_file, "r") as file:
        norm_params = dict(
            [
                (group.name.split("/")[-1], (np.array(group.get("mean")), np.array(group.get("standard"))))  # pyright: ignore[reportOptionalMemberAccess]
                for key in file.keys()
                if isinstance((group := file.get(key)), h5.Group)
            ]
        )
    return np.concatenate(
        [
            denormalize_quantity(quantity[:, 0:1], norm_params["x"]),  # pyright: ignore[reportArgumentType]
            denormalize_quantity(quantity[:, 1:2], norm_params["y"]),  # pyright: ignore[reportArgumentType]
        ],
        axis=1,
    )


# Normalizing data and using collate function from gnn_encoder.py
def preprocess_data(datasets: data, normalization_file: str) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    # Normalize data and labels
    norm_datasets = (
        data._make(
            [
                normalize_dataset(datasets.train, normalization_file),
                normalize_dataset(datasets.val, normalization_file),
                normalize_dataset(datasets.test, normalization_file),
            ]
        )
        if normalization_file != ""
        else data._make([datasets.train, datasets.val, datasets.test])
    )

    labelNames = ["x", "y"]

    return (
        DataLoader(norm_datasets.train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn),  # pyright: ignore[reportArgumentType]
        DataLoader(norm_datasets.val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn),  # pyright: ignore[reportArgumentType]
        DataLoader(norm_datasets.test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn),  # pyright: ignore[reportArgumentType]
        labelNames,
    )


# Assemble subplots and save directly or using get_img_from_matplotlib() from helper file
def create_plot_or_image(subplots: int, create_subplot: Callable, plot_folder: str, name: str, epoch: int, writer: SummaryWriter | None):
    fig = plt.figure(figsize=(16, 7.5))

    for index in range(subplots):
        create_subplot(index)

    plt.tight_layout()
    if plot_folder != "":
        fig.savefig(os.path.join(plot_folder, f"{name}_epoch_{epoch}.png"))
    if writer is not None:
        writer.add_image(name, get_img_from_matplotlib(fig), global_step=epoch, dataformats="NCHW")
    plt.close(fig)


# All evaluation of the trained model
def evaluate_and_plot(
    model: nn.Module,
    test_loader: DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor],
    device: torch.device,
    label_names: list[str] = [],
    writer: SummaryWriter | None = None,
    epoch: int = 0,
    plot_folder: str = "",
    normalization_file: str = "",
):
    """
    Evaluate the model and plot the results.

    This function evaluates the model on the test dataset and generates scatter plots
    and histograms to visualize the predictions and their distributions.
    The plots are logged to TensorBoard.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to evaluate.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    loss_function : callable
        Loss function used to compute the loss.
    device : torch.device
        Device on which to perform computations (e.g., 'cpu' or 'cuda').
    label_names : list[str]
        Names of the labels for plotting purposes.
    writer : SummaryWriter
        TensorBoard writer for logging.
    epoch : int
        Current epoch number (for logging purposes).
    plot_folder : str
        Folder to save the plots.
    """
    # Use evaluation function from helper file
    normalized_predictions, normalized_true_labels, _, _ = evaluate_model(model, test_loader, loss_function, device)

    # Denormalize x and y and concatenate for predictions and true labels
    if normalization_file != "":
        pred_mean = denormalize_dataset(normalized_predictions, normalization_file)
        all_true_labels = denormalize_dataset(normalized_true_labels, normalization_file)
    else:
        pred_mean = normalized_predictions
        all_true_labels = normalized_true_labels

    # Scatter plots for predictions
    def scatter_subplot(j: int) -> None:
        plt.subplot(1, 2, j + 1)
        plt.scatter(all_true_labels[:, j], y=pred_mean[:, j], s=6, alpha=0.2)
        plt.plot(
            [all_true_labels[:, j].min().item(), all_true_labels[:, j].max().item()],
            [all_true_labels[:, j].min().item(), all_true_labels[:, j].max().item()],
            c="black",
            linestyle="dashed",
            label="Perfect prediction",
        )
        plt.xlabel("true " + label_names[j])
        plt.ylabel("predicted " + label_names[j])
        plt.legend()

        return

    # Resolution plots
    def resolution_subplot(j: int) -> None:
        diff = all_true_labels[:, j] - pred_mean[:, j]

        plt.subplot(1, 2, j + 1)
        plt.hist(diff, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
        plt.axvline(diff.mean(), color="red", linestyle="dashed", linewidth=1)
        plt.xlabel(f"True - Predicted {label_names[j]}")
        plt.ylabel("Frequency")
        plt.title(f"Distribution for {label_names[j]}")

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

        return

    if plot_folder != "":
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

    create_plot_or_image(len(label_names), scatter_subplot, plot_folder, "scatterplot", epoch, writer)
    create_plot_or_image(len(label_names), resolution_subplot, plot_folder, "resolution", epoch, writer)

    return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the data and labels")
    parser.add_argument("-t", "--train", action="store_true", help="train model instead of using trained one in 'models/'")
    args = parser.parse_args()

    # Create the DataLoader for training, validation, and test datasets
    datasets = data._make(
        [
            awkward.from_parquet(os.path.join(DATA_PATH, "train.pq")),
            awkward.from_parquet(os.path.join(DATA_PATH, "val.pq")),
            awkward.from_parquet(os.path.join(DATA_PATH, "test.pq")),
        ]
    )
    inspect_datasets(datasets)

    # Important: We use the custom collate function to preprocess the data for GNN (see the description of the collate function for details)
    if not os.path.exists("outputs/"):
        os.makedirs("outputs/")
    normalization_file = "outputs/normalization_params.h5" if args.normalize else ""
    train_loader, val_loader, test_loader, label_names = preprocess_data(datasets, normalization_file)

    # Separately defined in gnn_encoder.py
    model = GNNEncoder(3, 128, 2)
    model_name = model._get_name()

    # Detect and use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # set up the loss function
    loss_fn = nn.MSELoss()  # Use MSE loss for regression

    def loss_function(inputs: torch.Tensor, labels: torch.Tensor, model: nn.Module) -> torch.Tensor:
        return loss_fn(model(inputs), labels)

    plot_kwargs = {"label_names": label_names, "test_loader": test_loader, "plot_folder": f"plots/{model_name}", "normalization_file": normalization_file}
    best_model_path = ("outputs/models/", f"{model_name}_best.pth")

    if args.train is True:
        # Initialize the Tensorboard writer and get current timestamp
        log_dir = f"logs/{model_name}/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        writer = SummaryWriter(log_dir)

        # Call the training function
        train_losses, val_losses, best_model = train_model(
            model,
            train_loader,
            val_loader,
            loss_function,
            learning_rate,
            num_epochs,
            patience,
            device,
            tensorboard_writer=writer,
            plot_fn=evaluate_and_plot,
            plot_interval=5,
            plot_kwargs=plot_kwargs,
        )

        # Final evaluation on the test dataset
        model.load_state_dict(best_model)  # Load the best model

        # Save the best model to the "models" directory
        if not os.path.exists(best_model_path[0]):
            os.makedirs(best_model_path[0])
        torch.save(best_model, best_model_path[0] + best_model_path[1])

    else:
        # Load the best model from the "models" directory
        model.load_state_dict(torch.load(best_model_path[0] + best_model_path[1], map_location=device, weights_only=True))

    model.to(device)

    # plot final model
    evaluate_and_plot(
        model,
        test_loader,
        loss_function,
        device,
        label_names=label_names,
        epoch=num_epochs,
        plot_folder=plot_kwargs["plot_folder"],
        normalization_file=plot_kwargs["normalization_file"],
    )

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
