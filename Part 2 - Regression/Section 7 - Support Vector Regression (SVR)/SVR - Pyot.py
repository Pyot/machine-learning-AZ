#SVR
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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()#tlumaczone s.7 w.60 8:23
sc_y = StandardScaler()
X = sc_X.fit_transform(X)#dlaczeto fit
y = sc_y.fit_transform(y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Prediciting a new result with Polymonila Regression S.5 w.52
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#Visualising ht SVR Regression results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth of Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising ht SVR Regression results hight resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth of Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

