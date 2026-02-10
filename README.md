# AI4EO-GEOL0069 Week 4 Assignment 

## Assignment Overview

The objective of this task is to classify altimetry echoes into two surface types (sea ice and leads) using a chosen unsupervised learning method (GMM) applied to Sentinel-3 altimetry data. We will process waveform echoes to distinguish between these two classes and compute the average echo shape and standard deviation for each class in order to characterise their statistical and physical differences. Finally we will quantify our echo classifications against the ESA official classification using a confusion matrix.
The classification workflow will build upon the provided notebook Chapter1_Unsupervised_Learning_Methods.ipynb.

## Installations

* Mounting Google Drive on Google Colab
  ```sh
  from google.colab import drive
  drive.mount('/content/drive')
  ```
* Required Python packages installed for this assignment:
* Using pip install:
  ```sh
  !pip install rasterio
  ```
  Reading and handling raster data
  ```sh
  !pip install netCDF4
  ```
  Reading Sentinel-3 altimetry files in NetCDF format
  ```sh
  !pip install msalign
  ```
  Pre-processing and aligning waveform echoes.
  ```sh
  !pip install scikit-learn (GaussianMixture, confusion_matrix, classification_report)
  ```
  Provides GMM for unsupervised classification and evaluation metrics
  
* Required Python packages usually pre-installed for this assignment:
  ```sh
  numpy
  ```
  Array operations and numerical computations
  ```sh
  matplotlib.pyplot
  ```
  Plotting echoes and visualizing results
  ```sh
  scipy.interpolate.interp1d
  ```
  Interpolating waveform echoes for alignment and averaging

  ## Context

Distinguishing sea ice from leads is important for several reasons: monitoring changes in polar ice coverage for climate research, improving navigation safety in polar regions, and supporting environmental and ecological studies. Understanding where open water occurs within sea ice helps scientists track melting, energy exchange between the ocean and atmosphere, and habitat availability for marine life.

Altimetry satellites, such as Sentinel-3, send radar signals toward the Earth's surface and record the reflected signals, known as echoes. These echoes carry detailed information about surface properties: differences in material, roughness, and elevation alter the shape and strength of the returned signal, making it possible to distinguish between surfaces like sea ice and leads.

By analyzing these echoes with unsupervised learning methods like the Gaussian Mixture Model (GMM), we can classify surface types, compute their average echo shapes and standard deviations, and compare the results to ESA official classifications. This provides a robust method for monitoring polar environments using satellite altimetry.

Below are links to European Space Agency (ESA) pages on Copernicus and Sentinel-3, as well as further information on the importance of understanding sea ice leads
* Copernicus - (https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Introducing_Copernicus)
* Sentinel-3 - (https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-3/Introducing_Sentinel-3)
* Sea ice Leads - (https://www.dkrz.de/en/communication/galerie/Vis/icesheet/sea-ice-leads)
 
 ## Introduction to Unsupervised Learning

Unsupervised learning methods are used when labels are not available in the dataset. Instead of relying on pre-defined categories, these algorithms identify patterns or groupings based on the inherent structure of the data. In this assignment, we apply the Gaussian Mixture Model (GMM), which clusters the altimetry echoes based on the distribution of selected features.

<p align="center">
  <img src="UnsupervisedLearning.png" alt="Unsupervised Learning Overview" width="600">
</p>

## Gaussian Mixture Modeling (GMM) of Altimetry Echoes

The Gaussian Mixture Model (GMM) is an unsupervised learning algorithm used to identify clusters in data when labels are not available. Unlike K-Means, which assigns points to the nearest cluster center, GMM assumes that each cluster follows a Gaussian (normal) distribution. Each data point has a probabilistic membership in every cluster, which allows GMM to model overlapping clusters and clusters of different shapes and sizes.
In this assignment, GMM is applied to altimetry echoes to classify sea ice and leads based on features derived from the waveform.

### Why GMM?
* Flexible: Can handle clusters with different shapes, sizes, and orientations.
* Probabilistic: Provides the likelihood that a point belongs to each cluster.
* More appropriate than K-Means for altimetry echoes, where clusters are not perfectly spherical.
* Other unsupervised alternatives could include DBSCAN or hierarchical clustering, but GMM offers the best trade-off between interpretability and performance for this dataset.

### GMM Basic Code
```sh
 from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
```
<p align="center">
  <img src="GMM.png" alt="Gaussian Mixture Model Overview" width="600">
</p>

This visualization shows the results of a Gaussian Mixture Model (GMM) applied to a 2D dataset. Each colored cluster represents a Gaussian component identified by the model, where points are grouped based on probabilistic membership rather than hard boundaries. The larger grey markers indicate the estimated cluster means, highlighting the center of each Gaussian distribution.

#### Features Used for Clustering
The clustering model uses waveform-derived features that capture surface reflectivity, roughness, and signal variability:

- `sig_0` — Normalized backscatter amplitude (indicative of surface reflectivity)
- `PP` — Peakiness of the waveform (reflecting surface roughness)
- `SSD` — Stack standard deviation (representing variability across repeated echoes)

To evaluate and interpret the Gaussian Mixture Model (GMM) clustering results, scatter plots were generated for each pair of input features (`sig_0` vs `PP`, `sig_0` vs `SSD`, and `PP` vs `SSD`). Plotting feature pairs allows us to visually assess how well the clusters are separated in feature space and to understand which variables contribute most to distinguishing surface types.

<p align="center">
  <img src="sig_0pp.png" alt="sig_0 vs pp" width="600">
</p>

<p align="center">
  <img src="sig_0SSD.png" alt="sig_0 vs SSD" width="600">
</p>

<p align="center">
  <img src="PPSSD.png" alt="PP vs SSD" width="600">
</p>

These plots show that the purple cluster forms a distinct grouping, clearly separated from the yellow cluster across all feature combinations. This separation indicates that the GMM has successfully identified statistically and physically different surface scattering behaviours. The distinct cluster suggests differences in surface reflectivity (`sig_0`), roughness (`PP`), and signal variability (`SSD`), implying that the model is capturing meaningful variation rather than arbitrary groupings.

## Echo Classification: Leads vs Sea Ice

This section classifies radar echoes into leads (open water) and sea ice, then evaluates the classification by comparing it with the ESA official dataset. The analysis also computes the average waveform shape and standard deviation for each class to characterize their physical differences.

### Classification code
#### 1. Load Echo Data
The first step is loading the SAR echo data from the provided file path. The data is stored in a NetCDF file.
```sh
path = '/content/drive/MyDrive/Week 4/'
SAR_file = 'S3A_SR_2_LAN_SI_20190307T005808_20190307T012503_20230527T225016_1614_042_131______LN3_R_NT_005.SEN3'
SAR_data = Dataset(path + SAR_file + '/enhanced_measurement.nc')
```

#### 2. Model Setup and Fitting
We first classify the echoes into two categories (leads and sea ice - n_component=2) using a Gaussian Mixture Model (GMM).
```sh
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
```
####  3. Cluster Prediction
Each echo is then assigned to a cluster (0 or 1) based on the GMM’s learned distributions.
```sh
clusters_gmm = gmm.predict(data_cleaned)
```
#### 4. Cluster Summary
The number of echoes in each cluster is counted to summarise the classification results.
```sh
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Cluster counts:", class_counts)
```
#### Example output: {0: 8880, 1: 3315}
Note: Cluster 0 contains 8,880 echoes and cluster 1 contains 3,315 echoes. These clusters represent the two categories of echoes, which will be interpreted as leads or sea ice in further analysis.

## Echo Waveform Analysis: Raw, Mean, and Standard Deviation
In this section, we analyse the shape and variability of the echoes. This helps us understand how the echoes from leads and sea ice differ, both in their overall waveform and in the variability between individual echoes.

### Raw Waveform Plots
The raw waveform plots show the first 7 echo waveforms from each cluster (0 for sea ice or 1 for leads). This visualisation allows us to compare the variability and shape of the echoes within each category before any statistical summarisation. Plotting multiple echoes together highlights differences in timing, amplitude, and overall structure.
```sh
import matplotlib.pyplot as plt

# Create subplots with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot first 7 functions where clusters_gmm is equal to 0
functions_to_plot_0 = waves_cleaned[clusters_gmm == 0][:7]
for i, function in enumerate(functions_to_plot_0):
    axs[0].plot(function, label=f'Function {i+1}')

# Plot first 7 functions where clusters_gmm is equal to 1
functions_to_plot_1 = waves_cleaned[clusters_gmm == 1][:7]
for i, function in enumerate(functions_to_plot_1):
    axs[1].plot(function, label=f'Function {i+1}')

# Set titles
axs[0].set_title('Clusters GMM = 0: Variability in Sea Ice')
axs[1].set_title('Clusters GMM = 1: Variability in Leads')

# Set axis labels
for ax in axs:
    ax.set_xlabel('Radar Bin')
    ax.set_ylabel('Power')

# Add legends
axs[0].legend()
axs[1].legend()

# Adjust layout and show
plt.tight_layout()
plt.show()
```
<p align="center">
  <img src="ClusterGMM.png" alt="ClusterGMM" width="600">
</p>

### Results:
#### Cluster 0 (Sea Ice):
* The waveforms are highly variable and chaotic.
* Peaks occur at different points across the time axis.
* There is substantial variation between individual echoes, indicating that sea ice
* echoes are less consistent in shape.
#### Cluster 1 (Leads):
* The waveforms are more orderly and consistent.
* Peaks tend to occur at the same point across all echoes.
* Some variation in peak height remains, but the overall structure is much more uniform compared to sea ice, reflecting the more coherent nature of lead echoes.

### Interpretation:
The raw waveform comparison demonstrates that sea ice echoes are more chaotic, while lead echoes show more regularity, which aligns with the physical differences in surface structure between sea ice and open water.
