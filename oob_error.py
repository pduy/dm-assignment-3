# OOB error for random forest
import matplotlib.pyplot as plt
import numpy as np
import generate

from sklearn import cross_validation
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

RANDOM_STATE = 123
n_estimators = np.logspace( -15, 15, 100, base=2)

X, y , attri_names = generate.return_data()
# What does warm_start do?
ensemble_clfs = [
     # ("RandomForestClassifier, warm_start=True",
     #    RandomForestClassifier(warm_start=True, max_features=None,
     #                           oob_score=True,
     #                           random_state=RANDOM_STATE)),
      ("RandomForestClassifier, warm_start=False",
    	RandomForestClassifier(warm_start=False, max_features=None,
                           	   oob_score=True,
                           	   random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
error_cv = OrderedDict((label, []) for label, _ in ensemble_clfs)
# Range of `n_estimators` values to explore.
min_estimators = 0
max_estimators = 7
n_estimators = np.logspace(min_estimators, max_estimators, max_estimators+1, base = 2, dtype = int)

for label, clf in ensemble_clfs:
    for i in n_estimators:
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

        # Use 10-fold cv and record 1-accuracy for each estimator
        cv_error = 1-np.mean(cross_validation.cross_val_score(clf,X,y,scoring ='accuracy', cv = 10))
        error_cv[label].append((i, cv_error))


# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label+ 'OOB Error')
for label, clf_err in error_cv.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label+ 'Error Rate (under 10-fold cv)')
plt.title('RandomForest Classifier performance against number of trees')
plt.xlabel("n_estimators")
plt.ylabel("error rate")
plt.legend(loc="upper right")
plt.show()