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
    X = np.genfromtxt('X.csv', delimiter = ',')
    y = np.genfromtxt('y.csv', delimiter = ',')
    return X, y

def train_neural_network(X, y, sgd, decay, n_epochs, batch):
    model = Sequential()
    #model.add(Dense(output_dim = 50, W_regularizer = l2(decay), input_dim = 14, init = 'uniform'))
    model.add(Dense(output_dim = 512, input_dim = 14, init = 'uniform', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim = 2, init = 'uniform', activation = 'softmax' ))

    X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size = 0.2, random_state = 33)

    y_train_original = [int(y) for y in y_train]
    y_test_original = [int(y) for y in y_test]
    y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]

    model.compile(loss='binary_crossentropy', optimizer=sgd)
    
    #stopping_pointer = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    #model.fit(X_train, y_train, nb_epoch=10000, batch_size=1000, validation_split = 0.1, show_accuracy=True, callbacks = [stopping_pointer])
    model.fit(X_train, y_train, nb_epoch=n_epochs, batch_size=batch, validation_split = 0.1, show_accuracy=True)
    score = model.evaluate(X_test, y_test, batch_size=batch)
    y_predict = model.predict_classes(X_test, batch_size=batch)
    mis_classification_rate = (1.0 * np.count_nonzero(y_predict - y_test_original) / len(y_predict))

    return score, mis_classification_rate

def random_search(X, y, n_iter):
    momentums = [0, 0.09, 0.099, 0.5, 0.9, 0.95, 0.99]
    #decays = [0, 0.01, 0.1, 0.5, 1, 2, 5]
    rates = [0.5, 0.1, 0.01, 0.001, 0.0001]
    dropouts = [0.25, 0.5, 0.75]
    epochs = [100, 1000, 10000]
    batches = [100, 1000, 10000]
    
    min_error = min_score = min_rate = np.inf
    min_decay = 0
    min_momentum = 0
    min_dropout = 0
    min_epoch = 0
    min_batch = 0
    
    for i in range(n_iter):
        momentum = momentums[np.random.randint(7)]
        #decay = decays[np.random.randint(6)]
        rate = rates[np.random.randint(5)]
        dropout = dropouts[np.random.randint(3)]
        epoch = epochs[np.random.randint(3)]
        batch = batches[np.random.randint(3)]

        sgd = SGD(lr=rate, momentum = momentum, nesterov = True)
        score, mis_classification_rate = train_neural_network(X, y, sgd, 0.1, epoch, batch)

        if mis_classification_rate < min_error:
            min_score = score
            min_error = mis_classification_rate
            min_dropout = dropout
            min_momentum = momentum 
            min_rate = rate
            min_epoch = epoch
            min_batch = batch 

    print 'min_score: %s' % min_score
    print 'min_error: %s' % min_error
    print 'min_dropout: %s' % min_dropout 
    print 'min_momentum: %s' % min_momentum
    print 'min_rate: %s' % min_rate 
    print 'min_epoch: %s' % min_rate 
    print 'min_batch: %s' % min_batch

    return min_score, min_error, min_decay, min_momentum

def print_training_result(X, y):
    sgd = SGD(lr=0.001, nesterov = True)
    score, mis_classification_rate = train_neural_network(X, y, sgd, 0.01)
    print 'score = %s' % score
    print 'mis_classification_rate = %s' % mis_classification_rate

def main():
    X, y = load_data()

    min_score, min_error, min_decay, min_momentum = random_search(X, y, 400)
    #print_training_result(X, y)

if __name__ == '__main__':
    main()
