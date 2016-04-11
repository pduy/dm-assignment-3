from scipy.io.arff import loadarff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
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

def train_neural_network(X, y, sgd, decay):
    model = Sequential()
    model.add(Dense(output_dim = 50, W_regularizer = l2(decay), input_dim = 784, init = 'uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim = 10, init = 'uniform' ))
    model.add(Activation('softmax'))

    X_train, y_train = X[0:59999], y[0:59999]
    X_test, y_test = X[60000:69999], y[60000:69999] 
    y_train_original = [int(y) for y in y_train]
    y_test_original = [int(y) for y in y_test]
    y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]

    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    model.fit(X_train, y_train, nb_epoch=1000, batch_size=5000, validation_split=0.1, show_accuracy=True)
    score = model.evaluate(X_test, y_test, batch_size=10000)
    y_predict = model.predict_classes(X_test, batch_size=60)
    mis_classification_rate = (1.0 * np.count_nonzero(y_predict - y_test_original) / len(y_predict))

    return score, mis_classification_rate

def random_search(X, y, n_iter):
    momentums = [0.5, 0.9, 0.95, 0.99]
    decays = [0.01, 0.1, 0.5, 1, 2, 5]
    min_error = min_score = np.inf
    min_decay = 0
    min_momentum = 0
    
    for i in range(n_iter):
        momentum = momentums[np.random.randint(4)]
        decay = decays[np.random.randint(5)]

        sgd = SGD(lr=0.001, momentum = momentum, nesterov = True)
        score, mis_classification_rate = train_neural_network(X, y, sgd, decay)

        if mis_classification_rate < min_error:
            min_score = score
            min_error = mis_classification_rate
            min_decay = decay
            min_momentum = momentum 

    return min_score, min_error, min_decay, min_momentum

def main():
    X, y = load_data()
    #X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=.14, random_state=0)

    
    sgd = SGD(lr=0.001, momentum = 0.9, nesterov = True)
    #score, mis_classification_rate = train_neural_network(X, y, sgd, 0.1)
    min_score, min_error, min_decay, min_momentum = random_search(X, y, 7)
    #print 'score = %s' % score
    #print 'mis_classification_rate = %s' % mis_classification_rate

    print 'min_score: %s' % min_score
    print 'min_error: %s' % min_error
    print 'min_decay: %s' % min_decay
    print 'min_momentum: %s' % min_momentum

if __name__ == '__main__':
    main()
