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

#linear Regression
lr_model=LinearRegression()
lr_model.fit(X_train,y_train)
lr_predictions=lr_model.predict(X_test)
lr_mse=mean_squared_error(y_test,lr_predictions)

#print the Mean Squared Error(MSE) for Linear Regression
print(f'Linear Regression MSE: {lr_mse}')
