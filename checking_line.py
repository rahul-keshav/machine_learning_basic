import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error



variable_X=np.array([[1],[2],[3]])


variable_Y=np.array([3,2,4])

model=linear_model.LinearRegression()

model.fit(variable_X,variable_Y)


print("weight",model.coef_)
print("intersept",model.intercept_)