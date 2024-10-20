# GridironGenius_ModelA_TitansD

This project is designed to predict the performance of NFL players for fantasy football using a Random Forest model. The model uses data from the 2024-2025 season to classify whether a player will be a "BOOM" (>= 20 fantasy points) or a "BUST" (< 20 fantasy points). The dataset used includes various statistics of players, which are processed and scaled before being fed into the model.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Modeling](#modeling)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Predicting New Players](#predicting-new-players)
- [Notes](#notes)

## Project Overview
The `GridironGenius_ModelA_TitansD` is a machine learning project implemented in Python, using a Random Forest Classifier to predict the fantasy performance of players in the NFL. The goal is to classify player performances as either "BOOM" or "BUST" based on historical data.

## Data
The model uses a dataset stored in a CSV file, which is automatically downloaded from the following URL:
- [Dataset Link](https://raw.githubusercontent.com/SIMBL742/GridironGenius/refs/heads/main/dataset_2024-2025_TitansD_QBP_.csv)

The target variable is `Fantasy Points`, and the model predicts whether the points will reach or exceed 20.

## Modeling
The following steps are performed in this notebook:
1. **Import Libraries**: Includes libraries such as `pandas`, `scikit-learn`, and `StandardScaler`.
2. **Load Data**: Loads the player data from a CSV file.
3. **Define X and Y**: Separates the features (X) and the target variable (y).
4. **Split Train and Test Data**: Divides the data into training and testing sets.
5. **Scale the Features**: Applies feature scaling using `StandardScaler`.
6. **Train the Model**: Trains a `RandomForestClassifier` on the training set.
7. **Make Predictions**: Predicts player performance on the test set.

## Dependencies
- Python 3.x
- pandas
- scikit-learn

## Usage
To use this code, simply run the Python script or Jupyter notebook. The model will train on the provided dataset and make predictions based on the test set. 

## Predicting New Players
The script includes a function, `predict_fantasy_player()`, which allows users to predict the performance of a specific NFL player for a given week:
1. Enter the player's first and last name.
2. Enter the current week of the NFL season.
3. Input the player's average stats for the given week (suggested to use data from ESPN.com).
4. The model will predict if the player is likely to be a "BOOM" or "BUST" for that week.

This feature allows users to input custom player data and see predictions in real-time.

## Notes
- Ensure that the CSV file is formatted correctly with feature names matching those expected by the model.
- Modify the feature extraction in the `predict_fantasy_player()` function if the feature set changes.
- The `RandomForestClassifier` uses 5 estimators, but you can adjust the `n_estimators` parameter for different performance outcomes.
- This notebook was developed and tested in Google Colab, but it should also work in other Jupyter environments with the appropriate libraries installed.
