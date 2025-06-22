import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
dataset = pd.read_csv(r"C:\Users\PC\OneDrive\Desktop\MSc Data Analytics\Predictive Analytics and Machine Learning using Python\8 Assignments\Decision Tree\Sales_Segments_data.csv")

# Perform one-hot encoding for the 'Segment' column
dataset = pd.get_dummies(dataset, columns=['Segment'], drop_first=True)

# Remove outliers using the IQR method for numeric columns only
numeric_cols = dataset.select_dtypes(include=[np.number])  # Select only numeric columns
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

# Adjust condition for filtering the dataset
condition = ~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)
dataset_no_outliers = dataset[condition]

# Split the data into features (X) and target variable (y)
X = dataset_no_outliers.drop('Sales ($)', axis=1)
y = dataset_no_outliers['Sales ($)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Regressor
regressor = DecisionTreeRegressor()

# Train the model on the training data
regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Visualization
plt.figure(figsize=(12, 8))

# Plot actual sales for Corporate segment
plt.scatter(X_test.index[X_test['Segment_Corporate'] == 1], y_test[X_test['Segment_Corporate'] == 1], 
            color='blue', marker='o', label='Actual Corporate Sales')

# Plot predicted sales for Corporate segment
plt.scatter(X_test.index[X_test['Segment_Corporate'] == 1], y_pred[X_test['Segment_Corporate'] == 1], 
            color='blue', marker='x', label='Predicted Corporate Sales')

# Plot actual sales for Home Office segment
plt.scatter(X_test.index[X_test['Segment_Home Office'] == 1], y_test[X_test['Segment_Home Office'] == 1], 
            color='green', marker='o', label='Actual Home Office Sales')

# Plot predicted sales for Home Office segment
plt.scatter(X_test.index[X_test['Segment_Home Office'] == 1], y_pred[X_test['Segment_Home Office'] == 1], 
            color='green', marker='x', label='Predicted Home Office Sales')

plt.xlabel('Index')
plt.ylabel('Sales')
plt.title('Comparison of Actual and Predicted Sales by Segment')
plt.legend()
plt.show()
