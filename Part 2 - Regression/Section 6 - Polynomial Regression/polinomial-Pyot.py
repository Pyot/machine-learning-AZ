import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset rodzielamy independent od dependent 
#y to dana ktorą bedziemy chcieli przewidziec
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

#Spliting the data set into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)"""

#Feature Scaling- przyspisze prace algorytmu
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =  sc_X.fit_transform(X_train)
X_test =  sc_X.transform(X_test)"""# -*- 

#Fitting Linear Regressionto the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) #jak dziala fit

#Fitting Polynomil Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)#transform dlaczego?
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth of Bluff (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
#Visualising ht POlynomil Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth of Bluff (Polynomiar Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
#Predicting a new result with Linear Regressin Sekcja 5 wyklad 52
lin_reg.predict(6.5)

#Prediciting a new result with Polymonila Regression S.5 w.52
lin_reg_2.predict(poly_reg.fit_transform(6.5))