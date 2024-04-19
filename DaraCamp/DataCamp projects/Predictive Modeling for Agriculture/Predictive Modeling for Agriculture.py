# This script performs multiclass classification using logistic regression to predict the type of crop based on soil measures.
# It imports necessary libraries including pandas for data manipulation, matplotlib for visualization, seaborn for creating heatmaps, and scikit-learn for logistic regression modeling.
# The dataset containing soil measures for different crops is loaded from a CSV file.
# Missing values in the dataset are checked and found to be none.
# The unique types of crops are identified to understand the multi-class target variable.
# The dataset is split into training and testing sets using train_test_split from scikit-learn.
# For each soil measure feature (N, P, K, and pH), a logistic regression model is trained and evaluated using the F1-score.
# The correlation matrix is calculated for the soil measures, and a heatmap is created using seaborn to visualize the correlations.
# Based on the correlation analysis, the final features for the model are selected.
# The data is split again using the final features, and a logistic regression model is trained and evaluated on the testing set using the F1-score.



# All required libraries are imported here for you.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Check for missing values
crops.isna().sum()

# Check how many crops we have, i.e., multi-class target
crops.crop.unique()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    crops[["N", "P", "K", "ph"]],
    crops["crop"],
    test_size=0.2,
    random_state=42
)

# Train a logistic regression model for each feature
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
    )
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"F1-score for {feature}: {f1}")

# Calculate the correlation matrix
crops_corr = crops[["N", "P", "K", "ph"]].corr()

# Create a heatmap using seaborn
sns.heatmap(crops_corr, annot=True)
plt.show()

# Select the final features for the model
final_features = ["N", "K", "ph"]

# Split the data with the final features
X_train, X_test, y_train, y_test = train_test_split(
    crops[final_features],
    crops["crop"],
    test_size=0.2,
    random_state=42
)

# Train a new model and evaluate performance
log_reg = LogisticRegression(
    max_iter=2000, 
    multi_class="multinomial"
)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
model_performance = f1_score(y_test, y_pred, average="weighted")
