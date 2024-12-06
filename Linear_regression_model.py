from random import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Import dataset
df = pd.read_excel("data/data.xlsx")

# Independent and dependent features
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(x.shape)

# Creating the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Building and training the model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Inference
y_pred = model.predict(x_test)
print(y_pred)

# Making a prediction for a single data point
print(model.predict([[15, 40, 1000, 75]]))  # Ensure this input matches the feature count

# Evaluating the model
# R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Adjusted R-squared
k = x_test.shape[1]  # Number of features
n = x_test.shape[0]  # Number of observations
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
print(f"Adjusted R-squared: {adj_r2}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(6, 8))
plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.legend()
plt.show()

# Save the trained model to a .pkl file
joblib.dump(model, "model/model.pkl")
