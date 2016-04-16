from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import numpy as np

def grid_search(X_train, y_train, param_grid):
    clf = ExtraTreesClassifier()

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 10, scoring = 'roc_auc')
    grid_search.fit(X_train, y_train)
    
    scores = [score[1] for score in grid_search.grid_scores_]

    print ('param = %s' % grid_search.best_params_)
    print ('Grid Search score = %s' % grid_search.best_score_)

    return scores

def run():
    X = np.genfromtxt('X.csv', delimiter = ',')
    y = np.genfromtxt('y.csv', delimiter = ',')  

    n_estimators = [128, 512, 1024, 2048]
    criterions = ['gini', 'entropy']
    min_samples_splits = [2, 5, 10, 15, 20]
    max_depths = [1000, 5000, 10000, 20000]

    param_grid = {'n_estimators':n_estimators,
            'criterion':criterions,
            'min_samples_split':min_samples_splits,
            'max_depth':max_depths}

    scores = grid_search(X, y, param_grid)
    print scores

if __name__ == '__main__':
    run()
