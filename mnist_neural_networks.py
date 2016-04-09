from scipy.io.arff import loadarff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
import numpy as np
from sklearn import cross_validation as cv

def load_data():
    pixel = 'pixel'
    attributes = [(pixel+str(i)) for i in range(1, 785)]
    data, meta = loadarff('mnist_784.arff')
    data = np.array(data)
    X = np.array([data[attr] for attr in attributes]).transpose()
    y = np.array(data['class'])
    return X, y

def main():
    X, y = load_data()
    #X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=.14, random_state=0)
    X_train, y_train = X[0:59999], y[0:59999]
    X_test, y_test = X[60000:69999], y[60000:69999]
    y_train_original = [int(y) for y in y_train]
    y_test_original = [int(y) for y in y_test]
    y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(output_dim = 50, input_dim=784, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim = 10, init = 'uniform' ))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    model.fit(X_train, y_train, nb_epoch=500, batch_size=30, show_accuracy=True)
    score = model.evaluate(X_test, y_test, batch_size=60)
    y_predict = model.predict_classes(X_test, batch_size=60)
    print 'loss = %s' % score
    print 'mis-classification rate = %s' % (1.0 * np.count_nonzero(y_predict - y_test_original) / len(y_predict))

if __name__ == '__main__':
    main()
