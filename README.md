# Credit Card Default Prediction

## Overview
This project aims to predict the likelihood of credit card default using machine learning techniques. The dataset contains information about credit card clients, including demographic and financial features. The main goal is to build a predictive model using Support Vector Machine (SVM) and optimize its performance through hyperparameter tuning and dimensionality reduction.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project can be found at the UCI Machine Learning Repository:
- [Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls)

### Description of Features:
- **DEFAULT**: Whether the client defaulted on their payment (1 = Yes, 0 = No)
- **SEX**: Gender of the client (1 = Male, 2 = Female)
- **EDUCATION**: Education level (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)
- **MARRIAGE**: Marital status (1 = Married, 2 = Single, 3 = Others)
- **AGE**: Age of the client
- **PAY_0 to PAY_6**: Payment status for the past 6 months (1 = paid duly, 0 = payment delay for one month, -1 = payment delay for two months, etc.)

## Data Preprocessing

1. **Loading the Data**: The data is loaded from an Excel file using `pandas`.
2. **Handling Missing Values**: The dataset is checked for missing values and rows with missing education or marital status are filtered out.
3. **Downsampling**: To balance the dataset, the majority class (clients who did not default) is downsampled to match the number of defaulting clients.
4. **One-Hot Encoding**: Categorical variables are converted to a numerical format using one-hot encoding.

## Modeling

1. **Splitting the Data**: The dataset is split into training and test sets using `train_test_split`.
2. **Scaling**: The features are scaled using `StandardScaler`.
3. **Model Training**: A Support Vector Machine (SVM) model is trained on the scaled data.
4. **Hyperparameter Tuning**: The model is optimized using Grid Search with cross-validation to find the best hyperparameters.
5. **Dimensionality Reduction**: PCA is used to reduce the dimensionality of the dataset before retraining the model.

## Results

- The model's accuracy is evaluated using the test set.
- A confusion matrix is displayed to visualize the performance of the model.

## Visualization

- **Count Plot**: Displays the distribution of the target variable (default or not).
- **Scree Plot**: Shows the explained variance for each principal component.
- **Decision Boundary Visualization**: A contour plot visualizes the decision boundaries of the SVM model in the PCA-transformed space.
