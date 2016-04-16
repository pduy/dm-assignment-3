import numpy as np
import matplotlib.pyplot as plt
import generate
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

def resample_split(X, y, state):
    # Train index
    train_index = resample(range(0,len(X)), random_state = state)
    X_train = X[train_index]
    y_train = y[train_index]
    # Test are the rest
    test_index = [i for i in range(len(X)) if i not in train_index]
    X_test = [X[i] for i in range(len(X)) if i not in train_index]
    y_test = [y[i] for i in range(len(X)) if i not in train_index]  
    return X_train, y_train, X_test, y_test, test_index

def main():
    global n_trees
    n_trees =  np.logspace( 0, 7,8, base=2)
    estimators = []
    for n in n_trees:
        estimators.append((str(n)+'tree', RandomForestClassifier(n_estimators = int(n))))
        
    X, y , attri_names = generate.return_data()

    global n_train
    global bias 
    global var
    global auc
    n_repeat = 5 
    bias = []
    var = []
    auc = []

    # For each bagging estimators, run n_repeat times
    for n, (name, estimator) in enumerate(estimators):

        y_predict = np.zeros((len(X), n_repeat))
        occurence = np.zeros(len(X))
        auc_scores = []
        # index = np.zeros((len(X), n_repeat))
        # Store list for calculate AUC
        prediction = []
        true = []

        for i in range(n_repeat):
            X_train, y_train,  X_test, y_test, test_index = resample_split(X,y,i)
            for s in test_index:
                occurence[s] += 1

            estimator.fit(X_train, y_train)
            predict = estimator.predict(X_test)
            # Count for prediction of class 1
            for p in range(len(X_test)):
                y_predict[test_index[p],i] = predict[p]

            # Compute auc score for this iteration
            predict_proba = np.array(estimator.predict_proba(X_test))
            local_auc = roc_auc_score(y_test, predict_proba[:,1])
            auc_scores.append(local_auc)

        y_var = np.zeros(len(X))
        y_bias = np.zeros((len(X),1))
        for row in range(len(X)):

            n_predictions_1 = sum(y_predict[row])
            n_predictions_0 = occurence[row] - n_predictions_1
            
            if y[row]:      # true label == 1
                mis = n_predictions_0
            else: 
                mis = n_predictions_1

            if occurence[row]:
                weight = occurence[row] / n_repeat
                y_bias[row] = np.square(mis/ occurence[row]) * weight
                y_var[row] = 0.5 * (1 - (np.square(n_predictions_1 * 1.0 / occurence[row]) + np.square(n_predictions_0 * 1.0 / occurence[row])))
                y_var[row] *= weight

                # Stroe prediction for AUC values
                prediction.append(n_predictions_1/occurence[row])
                true.append(y[row])

            else:
                y_var[row] = 0
                y_bias[row] = 0

        # Sum over itterations
        bias.append(sum(y_bias)/len(X))
        var.append(sum(y_var)/len(X))
        #auc.append(roc_auc_score(true, prediction))
        auc.append(np.mean(auc_scores))

    line_graph(n_repeat)

def line_graph(n_repeat):
    # Plot var 
    plt.subplot(311)
    plt.title('Bagging Performances against number of trees' + '\n' + str(n_repeat) + ' iterations')
    plt.plot(range(len(n_trees)), var, label = 'var' )
    plt.xticks( range(len(n_trees)), n_trees)
    plt.legend(loc="upper right")
    # Plot bias
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

if __name__ == '__main__':
    main()

