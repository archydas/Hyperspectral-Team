# -*- coding: utf-8 -*-
"""
Calculate SNR for hyperspectral data using both spatial and spectral domain methods

@author: Morteza
"""

import numpy as np
import spectral
import matplotlib.pyplot as plt

# Load your hyperspectral image
input_folder = "C:/Users/Morteza/OneDrive/Desktop/PhD/dataN/Data"
base_name = "Seurat_BEFORE"
header_file = f"{input_folder}/{base_name}.hdr"
spectral_image = spectral.open_image(header_file)

# Access the data as a NumPy array
hyperspectral_data = spectral_image.load()
wavelengths = np.array([float(w) for w in spectral_image.metadata['wavelength']])

# Function for Spatial Domain SNR Calculation
def calculate_spatial_snr(hyperspectral_data):
    num_bands = hyperspectral_data.shape[2]
    spatial_snr_values = []

    for i in range(num_bands):
        band_data = hyperspectral_data[:, :, i]
        signal_mean = np.mean(band_data)
        noise_std = np.std(band_data)
        snr = signal_mean / noise_std if noise_std != 0 else 0
        spatial_snr_values.append(snr)

    return np.array(spatial_snr_values)

# Function for Spectral Domain SNR Calculation per pixel
def calculate_spectral_snr(hyperspectral_data):
    height, width, num_bands = hyperspectral_data.shape
    spectral_snr_values = np.zeros((height, width))

    # Calculate SNR for each pixel across all spectral bands
    for i in range(height):
        for j in range(width):
            pixel_spectrum = hyperspectral_data[i, j, :]
            signal_mean = np.mean(pixel_spectrum)
            noise_std = np.std(pixel_spectrum)
            spectral_snr_values[i, j] = signal_mean / noise_std if noise_std != 0 else 0

    return spectral_snr_values

# Calculate SNR for each method
spatial_snr_values = calculate_spatial_snr(hyperspectral_data)
spectral_snr_values = calculate_spectral_snr(hyperspectral_data)

# Plot Spatial Domain SNR
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, spatial_snr_values, label='Spatial Domain SNR', marker='o', linestyle='-', color='blue')
plt.title('Spatial Domain SNR Across Wavelengths')
plt.xlabel('Wavelength (nm)')
plt.ylabel('SNR')
plt.legend()
plt.grid(True)
plt.show()

# Plot Spectral Domain SNR as a heatmap
plt.figure(figsize=(10, 6))
plt.imshow(spectral_snr_values, cmap='viridis', aspect='auto')
plt.colorbar(label='Spectral Domain SNR')
plt.title('Spectral Domain SNR Across Pixels')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.show()

# Print average SNR values for additional insight
average_spatial_snr = np.mean(spatial_snr_values)
average_spectral_snr = np.mean(spectral_snr_values)
print(f"Average Spatial Domain SNR: {average_spatial_snr:.2f}")
print(f"Average Spectral Domain SNR: {average_spectral_snr:.2f}")
