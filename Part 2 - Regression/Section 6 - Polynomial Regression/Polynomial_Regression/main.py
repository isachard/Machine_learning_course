import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#fitting linear Regression to dataset
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x,y)


#fitting polynomial regression to dataset

from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree =4)
x_poly = polynomial_reg.fit_transform(x)

linear_reg_2 = LinearRegression()
linear_reg_2.fit(x_poly,y)


#visualing linear reg results
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_reg.predict(x), color = 'blue'  )
plt.title('Truth or Bluff (Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show

#visualing Poylnomial linear reg results
#x_grid = np.arrange(min(x), max(x),0.1)
#x_grid = x_grid.reshape(len(x_grid,1))
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_reg_2.predict(polynomial_reg.fit_transform(x)), color = 'blue'  )
plt.title('Truth or Bluff (Polynomoial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show


#predicitng new result with linear regression
#print(linear_reg.predict(np.array([6.5]).reshape(1, 1)))


#poly regression:
print(linear_reg_2.predict(polynomial_reg.fit_transform([6.5])))
