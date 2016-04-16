from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
import numpy as np

X = np.genfromtxt('X.csv', delimiter = ',')
y = np.genfromtxt('y.csv', delimiter = ',')

learning_rates = [0.1, 0.01, 0.001]
n_estimators = [256, 512, 1024, 2048]
max_depths = [1000, 10000, 15000, 20000]

max_auc = 0
best_rate = 0
best_estimator = 0
best_depth = 0

for rate in learning_rates:
    for n_estimator in n_estimators:
        for depth in max_depths:
            clf = XGBClassifier(learning_rate = rate, n_estimators = n_estimator, max_depth = depth)
            score = np.mean(cross_validation.cross_val_score(clf, X, y, cv=10, scoring='roc_auc'))

            print 'cross_val_score = %s ' % score 
            if score > max_auc:
                max_auc = score
                best_rate = rate
                best_estimator = n_estimator
                best_depth = depth

print 'best_rate = % s' % best_rate
print 'best_estimator = % s' % best_estimator
print 'best_depth = % s' % depth 
