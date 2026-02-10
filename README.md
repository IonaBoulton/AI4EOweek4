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
