# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:45:36 2020
@author: work
"""
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
"""from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)"""

# Fitting the Regression to the dataset
#regression model here


# Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)
# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# for higher resolutn and smoother curve
x_grid = np.array(min(x), max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1) #or 1,1
plt.scatter(X, y, color = 'red')
plt.plot(x_grid,regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


