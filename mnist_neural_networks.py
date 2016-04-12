from scipy.io.arff import loadarff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2
import numpy as np
from sklearn import cross_validation as cv
from keras.callbacks import EarlyStopping


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
    #model.add(Dense(output_dim = 50, W_regularizer = l2(decay), input_dim = 784, init = 'uniform'))
    model.add(Dense(output_dim = 50, input_dim = 784, init = 'uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dense(output_dim = 10, init = 'uniform' ))
    model.add(Activation('softmax'))

    X_train, y_train = X[0:59999], y[0:59999]
    X_test, y_test = X[60000:69999], y[60000:69999] 
    y_train_original = [int(y) for y in y_train]
    y_test_original = [int(y) for y in y_test]
    y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]

    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    stopping_pointer = EarlyStopping(monitor='loss', patience=0, verbose=0, mode='auto')
    model.fit(X_train, y_train, nb_epoch=10000, batch_size=1000, show_accuracy=True, callbacks = [stopping_pointer])
    score = model.evaluate(X_test, y_test, batch_size=600)
    y_predict = model.predict_classes(X_test, batch_size=600)
    mis_classification_rate = (1.0 * np.count_nonzero(y_predict - y_test_original) / len(y_predict))

    return score, mis_classification_rate

def random_search(X, y, n_iter):
    momentums = [0.09, 0.099, 0.5, 0.9, 0.95, 0.99]
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

    print 'min_score: %s' % min_score
    print 'min_error: %s' % min_error
    print 'min_decay: %s' % min_decay
    print 'min_momentum: %s' % min_momentum

    return min_score, min_error, min_decay, min_momentum

def print_training_result(X, y):
    sgd = SGD(lr=0.001, nesterov = True)
    score, mis_classification_rate = train_neural_network(X, y, sgd, 0.1)
    print 'score = %s' % score
    print 'mis_classification_rate = %s' % mis_classification_rate

def main():
    X, y = load_data()

    min_score, min_error, min_decay, min_momentum = random_search(X, y, 12)
    #print_training_result(X, y)

if __name__ == '__main__':
    main()
