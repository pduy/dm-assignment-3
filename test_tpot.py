from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import csv

X = np.genfromtxt('X.csv', delimiter = ',')
y = np.genfromtxt('y.csv', delimiter = ',')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

tpot = TPOT(generations=10)
#tpot.fit(X_train, y_train)
tpot.fit(X, y)
#print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline.py')
