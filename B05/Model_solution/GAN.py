import sys, os
from typing import Tuple, NamedTuple, Callable

import torch
from torch import Tensor, nn, optim  # For Optimizer
from torch.utils.data import DataLoader  # For Data Loader
from torch.utils.tensorboard import SummaryWriter  # For Tensor Board Visualisation
from torchvision import transforms, datasets, utils  # For Data Set, Image Transforms

# Hyperparameters
learning_rate = 3e-4
batch_size = 32  # Batch size
num_epochs = 250
log_step = 625  # the number of steps to log the images and losses to tensorboard

latent_dimension = 128  # 64, 128, 256
image_dimension = 28 * 28 * 1  # 784

# Select device to train on, we use GPU if available, otherwise we use CPU
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class Generator(nn.Module):
    """
    Generator Model
    """

    def __init__(self):
        super().__init__()
        # We normalize inputs to [-1, 1] to make outputs [-1, 1]
        hidden_dim = 256
        self.gen = nn.Sequential(nn.Linear(latent_dimension, hidden_dim), nn.LeakyReLU(0.01), nn.Linear(hidden_dim, image_dimension), nn.Tanh())

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    """
    Discriminator Model
    """

    def __init__(self):
        super().__init__()
        hidden_dim = 256
        self.disc = nn.Sequential(nn.Linear(image_dimension, hidden_dim), nn.LeakyReLU(0.01), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.disc(x)


def prepare_images(generator: Generator, real: Tensor):
    # Get the real images and flatten them
    # for simplicity, we flatten the image to a vector and to use simple MLP networks
    # 28 * 28 * 1 flattens to 784
    flat_real = real.view(-1, 784).to(device)
    batch_size = real.shape[0]

    # generate fake images
    noise = torch.randn(batch_size, latent_dimension).to(device)
    fake = generator(noise)

    return flat_real, fake


def train_discriminator(
    discriminator: Discriminator, criterion: nn.BCELoss, opt_discriminator: optim.Adam, real: Tensor, fake: Tensor
) -> Tuple[Tensor, torch.Size]:

    # Train Discriminator:
    def loss_calculation(image: Tensor, real: bool) -> Tuple[Tensor, torch.Size]:
        discriminated = discriminator(image)  # predict the discriminator output
        labels = torch.full_like(discriminated, 1.0 if real else 0.0)  # real images are labeled as 1, fake images are labeled as 0
        return (criterion(discriminated, labels), discriminated.size())  # calculate the loss

    # calculate the # average loss for real and fake images
    loss_discriminator_real, disc_shape = loss_calculation(real, real=True)
    loss_discriminator_fake, _ = loss_calculation(fake, real=False)
    loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2

    # now we update the weights of the discriminator by backpropagating the loss
    # through the discriminator and we use the optimizer to update the weights
    # the generator is not updated in this step
    discriminator.zero_grad()
    loss_discriminator.backward(retain_graph=True)
    opt_discriminator.step()

    return loss_discriminator, disc_shape


def train_generator(
    generator: Generator, opt_generator: optim.Adam, discriminator: Discriminator, criterion: nn.BCELoss, disc_shape: torch.Size, fake: Tensor
) -> Tensor:
    # we generate fake images and pass them through the discriminator
    # we do a little trick and modify the original objective function of
    # minimizing the probability of the discriminator predicting the fake images as fake
    # to maximizing the probability of the discriminator predicting the fake images as real
    # this leads to a faster training of the generator when it does not represent the real data well
    # this is a common trick in GANs
    # for more information see section 17.1.2 of the book Deep Learning by Bishop and Bishop
    loss_generator = criterion(discriminator(fake), torch.ones(disc_shape).to(device))
    generator.zero_grad()
    loss_generator.backward()
    opt_generator.step()  # we only update the weights of the generator

    return loss_generator


# Initialize the Tensorboard writer
Summary = NamedTuple("summaries", [("fake", SummaryWriter), ("real", SummaryWriter), ("loss", SummaryWriter)])
writer = Summary._make([SummaryWriter("logs/fake"), SummaryWriter("logs/real"), SummaryWriter("logs/loss")])


# Adding tensorboard logging for losses and generated images
def log_to_tensorboard(step: int, real: Tensor, fake: Tensor, loss_discriminator: Tensor, loss_generator: Tensor) -> None:
    with torch.no_grad():
        # make grid of pictures and add to tensorboard
        # Generate noise via Generator, we always use the same noise to see the progression
        writer.fake.add_image("Mnist Fake Images", utils.make_grid(fake.reshape(-1, 1, 28, 28), normalize=True), global_step=step)
        # Get real data
        writer.real.add_image("Mnist Real Images", utils.make_grid(real.reshape(-1, 1, 28, 28), normalize=True), global_step=step)

        writer.loss.add_scalar("Loss Discriminator", loss_discriminator, global_step=step)
        writer.loss.add_scalar("Loss Generator", loss_generator, global_step=step)

    # increment step
    return


# loop for sequential discriminator and generator training
def training_loop(
    loader: DataLoader,
    discriminator: Discriminator,
    generator: Generator,
    opt_discriminator: optim.Adam,
    opt_generator: optim.Adam,
    criterion: nn.BCELoss,
    fixed_noise: Tensor,
) -> None:
    step = 0
    print("Started Training and visualization...")
    # Training Loop
    for epoch in range(num_epochs):
        # loop over batches
        for batch_idx, (real, _) in enumerate(loader):
            flat_real, fake = prepare_images(generator, real)
            # Train the discriminator on real images vs. generated images
            loss_discriminator, disc_shape = train_discriminator(discriminator, criterion, opt_discriminator, flat_real, fake)
            # Train generator
            loss_generator = train_generator(generator, opt_generator, discriminator, criterion, disc_shape, fake)

            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)}, Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}",
                end="\r",
            )

            # Log the losses and example images to tensorboard
            if batch_idx % log_step == 0:
                log_to_tensorboard(step, flat_real, generator(fixed_noise), loss_discriminator, loss_generator)
                step += 1

    return


def main() -> int:
    # we define a transform that converts the image to tensor and normalizes it with mean and std of 0.5
    # which will convert the image range from [0, 1] to [-1, 1]
    myTransforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # the MNIST dataset is available through torchvision.datasets
    print("loading MNIST digits dataset")
    # let's create a dataloader to load the data in batches
    loader = DataLoader(datasets.MNIST(root="dataset/", transform=myTransforms, download=True), batch_size=batch_size, shuffle=True)

    # initialize networks and optimizers
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    optimizer = lambda parameters: optim.Adam(parameters, lr=learning_rate)

    # generate one batch of random noise that we'll use to visualize the progression of the generator
    fixed_noise = torch.randn((batch_size, latent_dimension)).to(device)

    # train model with Adam optimizer and Binary Cross Entropy Loss
    training_loop(loader, discriminator, generator, optimizer(discriminator.parameters()), optimizer(generator.parameters()), nn.BCELoss(), fixed_noise)

    return os.EX_OK


if __name__ == "__main__":
    sys.exit(main())
