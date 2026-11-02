# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:32:04 2024

@author: Morteza
"""

import spectral
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load ENVI hyperspectral image
img = spectral.open_image('C:/Users/Morteza/OneDrive/Desktop/PhD/New_Data/8cal_Seurat_AFTER.hdr')
hyperspectral_data = img.load()
hyperspectral_data = hyperspectral_data[:, :, 0:151]

# Get the shape of the hyperspectral image
rows, cols, bands = hyperspectral_data.shape
print(f"Image shape: {hyperspectral_data.shape}")

# Reshape data to 2D array (pixels x bands) for NNMF
reshaped_data = hyperspectral_data.reshape((rows * cols, bands))

# Define the number of endmembers (n)
n_endmembers = 5

# Apply Non-Negative Matrix Factorization (NNMF) for spectral unmixing
nmf = NMF(n_components=n_endmembers, init='random', random_state=42)
W = nmf.fit_transform(reshaped_data)  # Abundance matrix (pixels x n_endmembers)
H = nmf.components_                  # Endmember matrix (n_endmembers x bands)

# Reshape the abundance matrix back to the original image format (rows, cols, n_endmembers)
abundance_maps = W.reshape((rows, cols, n_endmembers))

# Classify the hyperspectral image based on the highest abundance for each pixel
classified_image = np.argmax(abundance_maps, axis=2)  # Get the index of the endmember with the highest abundance

# Define a colormap with exactly 5 distinct colors for the 5 endmembers
colors = ['red', 'green', 'blue', 'yellow', 'purple']  # You can choose other colors if you prefer
cmap = ListedColormap(colors)

# Visualize the classified image with only 5 colors
plt.figure(figsize=(10, 10), dpi=150)
plt.imshow(classified_image, cmap=cmap, vmin=0, vmax=n_endmembers - 1)  # Use vmin and vmax to limit to 5 colors
cbar = plt.colorbar(ticks=np.arange(n_endmembers))  # Show colorbar with 5 levels
cbar.set_label('Class (Endmember)')
plt.title('Classified Hyperspectral Image Based on Endmembers')
plt.show()

# Visualize the abundance map of each endmember in 3 rows and 2 columns with higher DPI
fig, axes = plt.subplots(3, 2, figsize=(10, 15), dpi=150)  # Increase DPI to 150 for higher resolution

for i in range(n_endmembers):
    row = i // 2  # Determine row position
    col = i % 2   # Determine column position
    axes[row, col].imshow(abundance_maps[:, :, i], cmap='viridis')
    axes[row, col].set_title(f'Abundance of Endmember {i + 1}')
    axes[row, col].axis('off')  # Remove axis labels for clarity

# If there's an empty subplot (e.g., 6th position), hide it
if n_endmembers % 2 != 0:
    fig.delaxes(axes[2, 1])  # Remove the last empty subplot

plt.tight_layout()
plt.show()

# Plot the extracted endmembers (spectral signatures) with higher DPI
plt.figure(figsize=(10, 6), dpi=150)  # Increase DPI to 150 for better resolution
for i in range(n_endmembers):
    plt.plot(H[i], label=f'Endmember {i + 1}')
plt.xlabel('Spectral Band')
plt.ylabel('Reflectance')
plt.title('Extracted Endmembers')
plt.legend()
plt.show()
