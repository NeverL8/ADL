import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from tqdm.auto import tqdm

# # This is a simple example of a diffusion model in 1D.


# generate a dataset of 1D data from a mixture of two Gaussians
# this is a simple example, but you can use any distribution
data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1, 2])),
    torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1.]))
)

dataset = data_distribution.sample(torch.Size([10000]))  # create training data set
dataset_validation = data_distribution.sample(torch.Size([1000])) # create validation data set
fig, ax = plt.subplots(1, 1)
# sns.histplot(dataset,  stat='density', )
ax.set_title('Training data distribution')
sns.kdeplot(dataset, ax=ax, color='blue', label='True distribution', linewidth=2)
ax.set_xlabel('Sample value')
ax.set_ylabel('Probability Density')
fig.tight_layout()
fig.savefig('plots/A03simple_example_training_data.png', dpi=300)
# plt.show()
plt.close(fig)

# we will keep these parameters fixed throughout
TIME_STEPS = 250
BETA = torch.tensor(0.02)
N_EPOCHS = 2000
BATCH_SIZE = 64
LEARNING_RATE = 0.8e-4

# precompute the coefficients for the diffusion process
alphas = torch.tensor([(1-BETA) ** (i + 1) for i in range(TIME_STEPS)], dtype=torch.float32)

# define the neural network that predicts the amount of noise that was
# added to the data
# this is a simple feedforward network with 2 hidden layers
# the input is the data and the time step
g = torch.nn.Sequential(
    torch.nn.Linear(2, 20),
    torch.nn.ReLU(),
    # torch.nn.Linear(20, 20),
    # torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

optim = torch.optim.AdamW(g.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

loss_history = []
training_loss_history = []
epochs = tqdm(range(N_EPOCHS))  # this makes a nice progress bar
for e in epochs: # loop over epochs
    g.train()
    # loop through batches of the dataset, reshuffling it each epoch
    indices = torch.randperm(dataset.shape[0])
    shuffled_dataset = dataset[indices]
    tmp_loss = 0
    for i in range(0, shuffled_dataset.shape[0] - BATCH_SIZE, BATCH_SIZE):
        optim.zero_grad()

        # here, implement algorithm 1 of the DDPM paper (https://arxiv.org/abs/2006.11239)

        # sample a batch of data
        x0 = shuffled_dataset[i:i + BATCH_SIZE]
        # sample a random time step
        t = torch.randint(0, TIME_STEPS, (BATCH_SIZE,))
        # sample noise from a standard normal distribution
        noise = torch.randn_like(x0)
        # compute the noisy data sample after t steps
        xt = torch.sqrt(alphas[t]) * x0 + torch.sqrt(1 - alphas[t]) * noise
        # compute the loss
        loss = ((g(torch.cat((xt.unsqueeze(1), t.unsqueeze(1)), dim=1)).squeeze() - noise) ** 2).mean()
        tmp_loss += loss.item()

        loss.backward()
        optim.step()
    tmp_loss /= (shuffled_dataset.shape[0] // BATCH_SIZE)
    training_loss_history.append(tmp_loss)

    # compute the loss on the validation set
    g.eval()
    with torch.no_grad():
        x0 = dataset_validation
        t = torch.randint(0, TIME_STEPS, (dataset_validation.shape[0],))
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alphas[t]) * x0 + torch.sqrt(1 - alphas[t]) * noise
        loss = ((g(torch.cat((xt.unsqueeze(1), t.unsqueeze(1)), dim=1)).squeeze() - noise) ** 2).mean()
        epochs.set_postfix(loss=loss.item())
        loss_history.append(loss.item())

fig, ax = plt.subplots(1, 1)
ax.plot(training_loss_history, label='Training loss')
ax.plot(loss_history, label='Validation loss')
ax.legend()
ax.set_yscale('log')
ax.set_ylabel('Validation Loss')
ax.set_xlabel('Training step')
ax.set_ylim(0.1, 1)
fig.tight_layout()
fig.savefig('plots/A03simple_example_loss.png', dpi=300)



def sample_reverse(g, count, steps=TIME_STEPS):
    """
    Sample from the model by applying the reverse diffusion process

    Here, implement algorithm 2 of the DDPM paper (https://arxiv.org/abs/2006.11239)

    Parameters
    ----------
    g : torch.nn.Module
        The neural network that predicts the noise added to the data
    count : int
        The number of samples to generate in parallel
    steps : int, optional
        The number of time steps in the reverse diffusion process. Default is `TIME_STEPS`.
    Returns
    -------
    x : torch.Tensor
        The final sample from the model
    sample_history : list of torch.Tensor
        A list containing the history of sampled tensors at each time step, starting from the 
        initial noise sample and ending with the final sample.
    """
    # sample from a standard normal distribution
    g.eval()
    x = torch.randn(count)
    sample_history = [x]
    for t in range(steps - 1, -1, -1):
        # compute the noise that was added to the data
        noise = g(torch.cat((x.unsqueeze(1), torch.tensor([t] * count).unsqueeze(1)), dim=1))[:, 0]
        # apply the reverse diffusion process
        mean = 1/(1-BETA) ** 0.5 * (x - BETA/torch.sqrt(1-alphas[t]) * noise)
        if t > 0:
            # sample from the normal distribution
            epsilon = torch.randn(count)
            x = mean + torch.sqrt(BETA) * epsilon
        else:
            # last step, no noise
            x = mean

        sample_history.append(x)
    return x, sample_history



samples, history = sample_reverse(g, 1000)
samples = samples.detach().numpy()
history = np.array([x.detach().numpy() for x in history])
# plot the samples
fig, ax = plt.subplots(1, 1)
bins = np.linspace(-10, 10, 50)
sns.kdeplot(dataset, ax=ax, color='blue', label='True distribution', linewidth=2)
sns.histplot(samples, ax=ax, bins=bins, color='red', label='Sampled distribution', stat='density')
ax.legend()
ax.set_xlabel('Sample value')
ax.set_ylabel('Sample count')
fig.tight_layout()
fig.savefig('plots/A03simple_example_samples.png', dpi=300)


for j in range(1, 20):
    fig, ax = plt.subplots(1, 1)
    for i in range(0, j):
        # plt.plot(t, c='C%d' % int(t[-1] > 0), alpha=0.1)
        ax.plot(history[:, i], c='C%d' % int(history[-1, i] > 0), alpha=0.5)
    ax.set_title('Sample history')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Sample value')
    ax.set_ylim(-6,6)
    fig.tight_layout()
    fig.savefig(f'plots/A03simple_example_sample_history_{j}.png', dpi=300)
    plt.close(fig)
