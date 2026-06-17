import numpy as np
from matplotlib import pyplot as plt

data = np.load("../data/4/spectra.npy")

# labels: mass, age, l_bol, dist, t_eff, log_g, fe_h
labels = np.load("../data/4/labels.npy")

correlation_matrix = np.corrcoef(labels, rowvar=False)
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Symmetrize
np.fill_diagonal(correlation_matrix, 1)  # Set diagonal to 1

# Feature names
feature_names = ['mass', 'age', 'l_bol', 'dist', 't_eff', 'log_g', 'fe_h', 'SNR']

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
cax = plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(cax)

# Add labels
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')

# Add feature names to the axis
plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)

# Add gridlines
plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

# Add values to cells
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                 ha='center', va='center', color='black')

# Show plot
plt.tight_layout()



# plot a few spectra
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(data[0], lw=1)
ax.set_title("Star 0")

plt.tight_layout()

plt.show()