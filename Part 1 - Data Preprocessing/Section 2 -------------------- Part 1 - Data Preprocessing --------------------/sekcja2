# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset =  pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
# pokazuje cala array
np.set_printoptions(threshold=np.inf)
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values= "NaN", strategy="mean", axis = 0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

#Encoding catgorical data Sekcja 2.12.CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#Dummy Encoder France 100 Geramny 010 Spain 001
onehotencoder =  OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

#Spliting the data set into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Feature Scaling- przyspisze prace algorytmu
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =  sc_X.fit_transform(X_train)
X_test =  sc_X.transform(X_test)
