# twi-lite

## Description
**twi-lite** is a Python package that aims to make analyzing Twitter user data more convenient. It aims to provide a template to create a machine learning pipeline and high-level building blocks that allows full customization of feature selection and model evaluation based on the .

## Main Features
* A ```Feature```class that stores basic user information, the hashtags and the retweets that user has used
* Transformation functions that filters the data based on frequency and represents it with a SciPy sparse matrix
* Dimension reduction using UMAP to reduce the feature matrix to 2 dimensions for easier visualizations
* A ```Cluster```class that stores evaluation metrics and the clustering model to be used
* An evaluation function that uses the silhouette score to tune the hyperparameter for the clustering algorithm
* A ```Pipeline```class that combines each individual step into a machine learning pipeline that takes in Twitter data and returns user clusters for the selected feature

## Dependencies
* ```NumPy```
* ```Pandas```
* ```SciPy```
* ```Sci-kit learn```
* ```UMAP-learn```