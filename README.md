# Breast Cancer Classification using Logistic Regression and KNN

This repository contains a machine learning model for classifying breast cancer as malignant or benign based on features extracted from cell nuclei. The model uses Logistic Regression and K-Nearest Neighbors (KNN) for classification and performs hyperparameter tuning using GridSearchCV.

## Dataset

The dataset used is 'data.csv'.  This file contains features extracted from images of cell nuclei, along with a diagnosis label ('M' for malignant, 'B' for benign). The features represent various characteristics of the cell nuclei.

## Data Preprocessing

The code performs the following preprocessing steps:

* **Loads the dataset:** Reads the 'data.csv' file into a Pandas DataFrame.
* **Handles missing values:** Uses `SimpleImputer` to replace missing values with the mean of each column.
* **Converts diagnosis labels:** Maps 'M' to 1 (malignant) and 'B' to 0 (benign).
* **Feature scaling:** Applies `StandardScaler` to standardize the features. This is crucial for algorithms like KNN and Logistic Regression that are sensitive to feature scaling.
* **Splits the dataset:** Divides the data into training and validation sets using `train_test_split` with stratification to maintain class proportions in both sets.

## Models

Two classification models are used:

* **Logistic Regression:** A linear model that predicts the probability of a sample belonging to a particular class. Hyperparameter tuning (regularization parameter 'C' and solver) is performed using `GridSearchCV`.
* **K-Nearest Neighbors (KNN):** A non-parametric method that classifies a data point based on the majority class among its k-nearest neighbors in the feature space. Hyperparameter tuning ('n_neighbors') is performed using `GridSearchCV`.

## Evaluation Metrics

The performance of both models is evaluated using the following metrics:

* **Accuracy:** The percentage of correctly classified instances.
* **Confusion Matrix:** A table showing the counts of true positive, true negative, false positive, and false negative predictions.
* **Classification Report:** Includes precision, recall, F1-score, and support for each class.
* **AUC-ROC:** Area under the Receiver Operating Characteristic curve, measuring the model's ability to distinguish between classes.

## Results

The code outputs the best hyperparameters found during the GridSearchCV process for both Logistic Regression and KNN, along with their respective performance metrics on the validation set. This allows for a direct comparison of the two models' performance.
The results demonstrate a comparable performance between Logistic Regression and KNN on this breast cancer classification task. While achieving similar accuracy scores, the models exhibit differences in other metrics, such as precision and recall for each class. This suggests that the choice of model might depend on the specific priorities of the application (e.g., minimizing false positives vs. minimizing false negatives).

## Getting Started

1. **Prerequisites:**  Make sure you have Python and the following libraries installed:
   ```bash
   pandas
   matplotlib
   seaborn
   scikit-learn

## Licence
Creative Commons Zero v1.0 Universal
