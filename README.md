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

![Unsupervised Learning Overview](UnsupervisedLearning.png)


