import numpy as import matplotlib.pyplot as plt Import pandas as pd import sklearn from sklearn.model_selection inport train_test split from sklearn.metrics import mean_squared error,r2_score from sklearn Import linear model from sklearn.datasets import load iris [12] x,y=load_iris(return_x_y=True) [z:]xx x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20, random_state=2) [14] x_train-np.array (x_train).reshape(-1,1) y_train-np.array(y_train).reshape(-1,1) x_test=np.array(x_test).reshape(-1,1) [15] classifier-linear model.LinearRegression() classifier.fit(x_train,y_train) y_pred=classiffer.predict(x_test) [16] mean_squared_error(y_test,y_pred) 0.051037102447046875 plt.scatter(x_test,y_test, color="black") plt.plot(x_test,y_pred, color-'blue') plt.show()