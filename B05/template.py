import torchvision
# For image transforms
from torchvision import transforms
# For DATA SET
import torchvision.datasets as datasets
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
# FOR DATA LOADER
from torch.utils.data import DataLoader
# FOR TENSOR BOARD VISUALIZATION
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard

# Hyperparameters
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
batchSize = 32  # Batch size
numEpochs = 100
logStep = 625  # the number of steps to log the images and losses to tensorboard

latent_dimension = 128 # 64, 128, 256
# for simplicity we will flatten the image to a vector and to use simple MLP networks
# 28 * 28 * 1 flattens to 784
# you are also free to use CNNs
image_dimension = 28 * 28 * 1  # 784

# we define a tranform that converts the image to tensor and normalizes it with mean and std of 0.5
# which will convert the image range from [0, 1] to [-1, 1]
myTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

class Generator(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dimension, ...) #TODO define a suitable hidden dimension
            ... #TODO define a suitable network architecture for the generator
            nn.Tanh(),  # It is helpful to use the tanh activation function to force the ouput into the [-1,1] range that our normalized images have.
        )

    def forward(self, x):
        return self.gen(x)



class Discriminator(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dimension, ...) #TODO define a suitable hidden dimension
            ... #TODO define a suitable network architecture for the discriminator,
        )

    def forward(self, x):
        return self.disc(x)


# initialize networks and optimizers
discriminator = Discriminator().to(device)
generator = Generator().to(device)
opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
opt_generator = optim.Adam(generator.parameters(), lr=lr)

# This is a binary classification task, so we use Binary Cross Entropy Loss
criterion = nn.BCELoss()


# Training Loop
step = 0
print("Started Training and visualization...")
for epoch in range(numEpochs):
    # loop over batches
    print()
    for batch_idx, (real, _) in enumerate(loader):
        # First we train the discriminator on real images vs. generated images

        # Get the real images and flatten them
        # for simplicity, we flatten the image to a vector and to use simple MLP networks
        # 28 * 28 * 1 flattens to 784
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Step 1) generate fake images
        noise = torch.randn(batch_size, latent_dimension).to(device)
        fake = generator(noise)

        # Step 2) Train Discriminator:
        # - predict the discriminator output for real images
        # - real images are labeled as 1
        # - calculate the loss for real images

        # - predict the discriminator output for fake images
        # -fake images are labeled as 0
        # -calculate the loss for fake images

        # -average the loss for real and fake images

        # - now upadate the weights of the discriminator by backpropagating the loss through the discriminator
        # the generator is not updated in this step
        # HINT: call the `backward` method of the discriminator with the argument `retain_graph=True` to keep the computational graph
        # this is necessary because we will use the same discriminator to train the generator

        # Train Generator:
        # Now train the generator by generating fake images and passing them through the discriminator
        # You can do a little trick and modify the original objective function of
        # "minimizing the probability of the discriminator predicting the fake images as fake"
        # to "maximizing the probability of the discriminator predicting the fake images as real"
        # this leads to a faster training of the generator when it does not represent the real data well
        # this is a common trick in GANs
        # for moer information see section 17.1.2 of the book Deep Learning by Bishop and Bishop

        # Todo:
        # - pass the fake images through the discriminator
        # - calculate the loss (by passing the output of the discriminator through the criterion with labels set to 1 (real images
        # - update the weights of the generator


        # print the progress
        print(f"\rEpoch [{epoch}/{numEpochs}] Batch {batch_idx}/{len(loader)} \ Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}", end="")

        # Log the losses and example images to tensorboard
        if batch_idx % logStep == 0:
            with torch.no_grad():
                # Generate noise via Generator, we always use the same noise to see the progression
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                # Get real data
                data = real.reshape(-1, 1, 28, 28)
                # make grid of pictures and add to tensorboard
                imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                imgGridReal = torchvision.utils.make_grid(data, normalize=True)

                # TODO: add the images and losses to tensorboard
                # HINT: use the SummaryWriter to add the images and scalars to tensorboard
                # HINT: use the `add_image` method to add the images to tensorboard
                # HINT: use the `add_scalar` method to add the losses to tensorboard

                # increment step
                step += 1