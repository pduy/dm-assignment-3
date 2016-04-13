from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier 
from matplotlib import pyplot as plt
import numpy as np

def plot_surface(clf, X, y,
        xlim=(-3, 4), ylim=(-3, 4), n_steps=250,
        subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
            np.linspace(ylim[0], ylim[1], n_steps))
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        if show:
            plt.show()

def main():
    X, y = datasets.make_moons(n_samples=200, shuffle=True, noise=0.1, random_state=None)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    for i in range(8):
        clf = RandomForestClassifier(n_estimators = 2**i)   
        clf.fit(X,y)
        plot_surface(clf, X, y)

if __name__ == '__main__':
    main()
