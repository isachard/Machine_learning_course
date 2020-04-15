import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



#importing dataset

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
"""


#encoding categorical data using dummy encoding
#labelencoder_x = LabelEncoder()
#x[:,0] = labelencoder_x.fit_transform(x[:,0])
#onehotenco = OneHotEncoder()
#x = onehotenco.fit_transform(x).toarray()

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

"""

#Splitting the dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1/3, random_state = 0 )

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)



#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualizing the training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train,  regressor.predict(x_train), color = 'blue') 
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#Visualizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train,  regressor.predict(x_train), color = 'blue') 
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()