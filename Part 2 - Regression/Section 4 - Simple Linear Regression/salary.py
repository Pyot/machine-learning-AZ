# -*- coding: utf-8 -*-
# Simple le

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Spliting the data set into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

#Feature Scaling- przyspisze prace algorytmu
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =  sc_X.fit_transform(X_train)
X_test =  sc_X.transform(X_test)"""

#Fitting Simple LInear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results (O CO TUTAJ CHODZI)
y_pred = regressor.predict(X_test)

#visutalizatin Training set resoult

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()


#visutalizatin test set resoult

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()