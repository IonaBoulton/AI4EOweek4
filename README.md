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
  ```sh
  !pip install netCDF4
  ```
  ```sh
  !pip install msalign
  ```
  ```sh
  !pip install scikit-learn
  ```
* Required Python packages usually pre-installed for this assignment:
  ```sh
  numpy
  ```
  ```sh
  matplotlib.pyplot
  ```
  ```sh
  scipy.interpolate.interp1d
  ```
