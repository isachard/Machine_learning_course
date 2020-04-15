#Multiple linear Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
#depent variables -> profit
#ind RD spend, marketing etc

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Enconding categorical Data
#Encoding the Independent variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)

#avoiding dummy variable trap
x = x[:,1:]



#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0 )


#Fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualizing the test set results
plt.scatter(x_train[:,[3]], y_train, color = 'red')
plt.plot(x_train,  regressor.predict(x_train), color = 'blue') 
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Building the optimal model using Backward elimination
import statsmodels.regression.linear_model as sm
x = np.append(arr = np.ones((50, 1)).astype(int),values = x, axis = 1)

#x_opt = x[:, [0,1,2,3,4,5]] first run 
x_opt = x[:, [0,3]]
x_opt = np.array(x_opt, dtype=float)

regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
 


"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Visualizing the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train,  regressor.predict(x_train), color = 'blue') 
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#Visualizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,  regressor.predict(x_tr ain), color = 'blue') 
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()"""