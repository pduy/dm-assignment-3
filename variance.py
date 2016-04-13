import numpy as np
import matplotlib.pyplot as plt
import generate

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.utils import resample
from sklearn import metrics

def resample_split(state):
	# Train index
	train_index = resample(range(0,len(X)), random_state = state, n_samples = n_test)
	X_train = X[train_index]
	y_train = y[train_index]
	# Test are the rest
	test_index = [i for i in range(len(X)) if i not in train_index]
	X_test = [X[i] for i in range(len(X)) if i not in train_index]
	y_test = [y[i] for i in range(len(X)) if i not in train_index]	
	return X_train, y_train, X_test, y_test, test_index


n_trees =  np.logspace( 0, 7,8, base=2)
estimators = []
for n in n_trees:
	estimators.append((str(n)+'tree', RandomForestClassifier(n_estimators = int(n))))
	
X, y , attri_names = generate.return_data()

n_test = 1000
n_repeat = 100 # number of repeats for getting bias, variance, ...
bias = []
var = []
auc = []

# For each bagging estimators, run n_repeat times
for n, (name, estimator) in enumerate(estimators):

	y_predict = np.zeros((len(X), n_repeat))
	index = np.zeros((len(X), n_repeat))
	# Store list for calculate AUC
	prediction = []
	true = []

	for i in range(n_repeat):
		X_train, y_train,  X_test, y_test, test_index = resample_split(i)
		for s in test_index:
			index[s,i] += 1 

		estimator.fit(X_train, y_train)
		predict = estimator.predict(X_test)
		# Count for prediction of class 1
		for p in range(len(X_test)):
			y_predict[p,i] += predict[p]

	y_var = np.zeros((len(X),1))
	y_bias = np.zeros((len(X),1))
	for row in range(len(X)):

		y1 = sum(y_predict[row])
		y0 = sum(index[row])-y1
		
		if y[row]:		# true label == 1
			mis = y0
		else: 
			mis = y1

		if sum(index[row]):
			weight = sum(index[row])/ n_test
			y_bias[row] = np.square(mis/ sum(index[row])) * weight
			y_var[row] = 1- 0.5 * (np.square(y1/sum(index[row])) + np.square(y0/ sum(index[row])))
			y_var[row] *= weight

			# Stroe prediction for AUC values
			prediction.append(y1/sum(index[row]))
			true.append(y[row])


		else:
			y_var[row] = 0
			y_bias[row] = 0

	# Sum over itterations
	bias.append(sum(y_bias)/len(X))
	var.append(sum(y_var)/len(X))

	fpr, tpr, thresholds = metrics.roc_curve(true, prediction)
	auc.append(metrics.auc(fpr, tpr))

# f, axarr = plt.subplots(2, sharex=True)
# axarr[0].plot(range(len(n_trees)), var, label = 'var' )
# axarr[0].plot(range(len(n_trees)), bias, label = 'bias')
# axarr[0].set_title('Bagging Performances against number of trees')
# axarr[1].plot(range(len(n_trees)), auc, label = 'auc')

# axarr[1].xticks( range(len(n_trees)), n_trees)
# plt.legend(loc="upper right")
# plt.show()

# Plot var & bias
plt.subplot(311)
plt.title('Bagging Performances against number of trees')
plt.plot(range(len(n_trees)), var, label = 'var' )
plt.xticks( range(len(n_trees)), n_trees)
plt.legend(loc="upper right")
# Plot auc
plt.subplot(312)
plt.plot(range(len(n_trees)), bias, label = 'bias', color = 'green')
plt.xticks( range(len(n_trees)), n_trees)
plt.legend(loc="upper right")
# Plot auc
plt.subplot(313)
plt.plot(range(len(n_trees)), auc, label = 'auc', color = 'red')
plt.xticks( range(len(n_trees)), n_trees)
plt.legend(loc="upper right")
plt.show()