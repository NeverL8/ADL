import copy
import io
from matplotlib.figure import Figure
import numpy as np
import os, sys
from PIL import Image
import time
from typing import Callable, Any

import torch
from torch.nn import Module
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor, Module], torch.Tensor],
    learning_rate: float,
    num_epochs: int,
    patience: int,
    device: torch.device,
    tensorboard_writer: SummaryWriter | None = None,
    plot_interval: int = 1,
    plot_fn: Callable | None = None,
    plot_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Trains a given model using the provided training and validation data loaders, loss function, and optimizer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        loss_function : callable
            Loss function to be used for training.
        learning_rate : float
            learning rate
        num_epochs : int
            Number of epochs to train the model.
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        device : torch.device
            Device on which to perform training (e.g., 'cpu' or 'cuda').
        tensorboard_writer : torch.utils.tensorboard.SummaryWriter, optional
            TensorBoard writer for logging training progress. Default is None.
        plot_interval : int, optional
            Interval for logging training progress. Default is 1 (log every epoch).
        plot_fn : callable, optional
            Function for plotting training progress. Default is None.
        plot_kwargs : dict, optional
            Additional arguments for the plotting function. Default is None.

        Returns
        -------
        tuple
            A tuple containing two lists:
            - train_losses (list of float): List of average training losses for each epoch.
            - val_losses (list of float): List of average validation losses for each epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model: dict = {}

    for epoch in range(num_epochs):
        start_time = time.time()  # Start the timer for this epoch

        # Training phase
        model.train()
        total_train_loss = 0.0
        for step, (batch_data, batch_labels) in enumerate(train_loader):

            ## move to device

            batch_labels = batch_labels.to(device)

            if type(batch_data) == torch.Tensor:
                batch_data = batch_data.to(device)
            else:
                batch_data[0] = batch_data[0].to(device)
                # batch_data[1]=batch_data[1].to(device)

            optimizer.zero_grad()
            loss = loss_function(batch_data, batch_labels, model)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Print progress every 10th step, updating the same line
            if (step + 1) % 10 == 0:
                sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                sys.stdout.flush()

        sys.stdout.write("\n")  # Move to the next line after the epoch

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for step, (batch_data, batch_labels) in enumerate(val_loader):
                ## move to device
                batch_labels = batch_labels.to(device)

                if type(batch_data) == torch.Tensor:
                    batch_data = batch_data.to(device)
                else:
                    batch_data[0] = batch_data[0].to(device)
                    # batch_data[1]=batch_data[1].to(device)

                val_loss = loss_function(batch_data, batch_labels, model)

                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print epoch summary
        epoch_time = time.time() - start_time  # Calculate epoch time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = copy.copy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # TensorBoard logging
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("Train Loss", avg_train_loss, epoch + 1)
            tensorboard_writer.add_scalar("Validation Loss", avg_val_loss, epoch + 1)

            # Log training and validation losses every log_interval epochs
            if plot_fn is not None and (epoch + 1) % plot_interval == 0:
                # Plotting
                assert plot_kwargs is not None
                assert "test_loader" in plot_kwargs.keys()
                print("plotting intermediate training state")
                plot_fn(
                    model,
                    plot_kwargs["test_loader"],
                    loss_function,
                    device,
                    label_names=plot_kwargs["label_names"],
                    writer=tensorboard_writer,
                    epoch=epoch,
                    plot_folder=plot_kwargs["plot_folder"],
                )

    return np.array(train_losses), np.array(val_losses), best_model


def evaluate_model(
    model: Module, test_loader: DataLoader, loss_function: Callable, device: torch.device
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Evaluate the given model on the test dataset.

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


    Returns
    -------
    all_predictions : numpy.ndarray
        Array of denormalized predictions made by the model.
    all_true_labels : numpy.ndarray
        Array of denormalized true labels from the test dataset.
    """
    model.eval()
    total_test_loss = 0.0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        first_batch_data = torch.Tensor()
        first_batch_labels = torch.Tensor()

        for batch_index, (batch_data, batch_labels) in enumerate(test_loader):

            batch_labels = batch_labels.to(device)
            if type(batch_data) == torch.Tensor:
                batch_data = batch_data.to(device)
            else:
                batch_data[0] = batch_data[0].to(device)

            predictions = model(batch_data)

            test_loss = loss_function(batch_data, batch_labels, model)

            total_test_loss += test_loss.item()
            all_predictions.append(predictions.cpu())
            all_true_labels.append(batch_labels.cpu())

            if batch_index > 0:
                continue
            first_batch_data = batch_data
            first_batch_labels = batch_labels

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    return torch.cat(all_predictions).numpy(), torch.cat(all_true_labels).numpy(), first_batch_data, first_batch_labels


def get_img_from_matplotlib(fig: Figure) -> torch.Tensor:
    """
    Convert a Matplotlib figure to a PyTorch tensor for logging
    in TensorBoard.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure to convert.
    Returns
    -------
    torch.Tensor
        A PyTorch tensor representing the image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg")
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image


def normalize_quantity(qty, norm_params: tuple[float, float] | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    if norm_params is None:
        qty_mean = np.mean(qty)
        qty_std = np.std(qty)
    else:
        qty_mean, qty_std = norm_params

    return (qty - qty_mean) / qty_std, (qty_mean, qty_std)  # pyright: ignore[reportReturnType]


def denormalize_quantity(coordinate: np.ndarray, norm_params: tuple[float, float]) -> np.ndarray:
    mean, std = norm_params
    return coordinate * std + mean
