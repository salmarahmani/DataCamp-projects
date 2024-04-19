# This script performs logistic regression on a dataset containing car insurance information to predict the outcome of car insurance claims.
# It imports necessary modules including pandas for data manipulation, numpy for numerical computations, and statsmodels for statistical modeling.
# The dataset "car_insurance.csv" is read into a DataFrame called 'cars'.
# Missing values in the 'credit_score' and 'annual_mileage' columns are filled with the mean of their respective columns.
# Logistic regression models are created for each feature individually to predict the outcome variable 'outcome'.
# Confusion matrices are computed for each model to calculate accuracy.
# The feature with the highest accuracy is identified, and its name along with the corresponding accuracy are stored in a DataFrame called 'best_feature_df'.



# Import required modules
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

# Read in dataset
cars = pd.read_csv("car_insurance.csv")

# Check for missing values
cars.info()

# Fill missing values with the mean
cars["credit_score"].fillna(cars["credit_score"].mean(), inplace=True)
cars["annual_mileage"].fillna(cars["annual_mileage"].mean(), inplace=True)

# Empty list to store model results
models = []

# Feature columns
features = cars.drop(columns=["id", "outcome"]).columns

# Loop through features
for col in features:
    # Create a model
    model = logit(f"outcome ~ {col}", data=cars).fit()
    # Add each model to the models list
    models.append(model)

# Empty list to store accuracies
accuracies = []

# Loop through models
for feature in range(0, len(models)):
    # Compute the confusion matrix
    conf_matrix = models[feature].pred_table()
    # True negatives
    tn = conf_matrix[0,0]
    # True positives
    tp = conf_matrix[1,1]
    # False negatives
    fn = conf_matrix[1,0]
    # False positives
    fp = conf_matrix[0,1]
    # Compute accuracy
    acc = (tn + tp) / (tn + fn + fp + tp)
    accuracies.append(acc)

# Find the feature with the largest accuracy
best_feature = features[accuracies.index(max(accuracies))]

# Create best_feature_df
best_feature_df = pd.DataFrame({"best_feature": best_feature,
                                "best_accuracy": max(accuracies)},
                                index=[0])
best_feature_df
