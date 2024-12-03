from random import random

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
# Import dataset
df = pd.read_excel("data/data.xlsx")

#Independent and,dependent  features
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(x.shape)

#Creating the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random=42)

#Building and training the model
model = LinearRegression()

#Train the model
model.fit(x train, y_train)

#inference
y_pred = model.predict(x_test)
print(y_pred)
#Making the prediction a single data point with AT = 15, V =40, AP=
print(model.predict([[15, 40, 1000, 75]]))

#Evaluating the model
#R-squared
r2 = r2_score(y_test, y_pred)
print(r2)

#Adjusted-squared
k = x_test.shape=[1]
n = x_test.shape=[0]
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
print(adj_r2)

#Scatter plot of actual vs.predicted values
plt.figure(figsize =(6,8))
#plot actual.vs predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.xlabel("Actual values")
plt.ylabel("predicted values")
plt.title("Actual vs.predicted values")
plt.show()

#save the trained model to a .pkl file
joblib.dump(model,"model/model.pkl")