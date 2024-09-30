import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = load_iris()
X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Multiple linear Regression
mlr_model=LinearRegression()
mlr_model.fit(X_train,y_train)
mlr_predictions=mlr_model.predict(X_test)
mlr_mse=mean_squared_error(y_test,mlr_predictions)

#print the Mean Squared Error(MSE) for Linear Regression
print(f'Multiple Linear Regression MSE: {mlr_mse}')
