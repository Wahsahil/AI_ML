from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt 
x=np.array([2,4,6,8,10]).reshape(-1,1)
y=np.array([1,2,3,4,5])
plt.scatter(x,y)
model = LinearRegression()
model.fit(x,y)
intercept = model.intercept_
coeff = model.coef_
y_pred = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()