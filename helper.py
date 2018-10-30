import scipy,numpy as np;
import copy

def sigmoid(X):

    return scipy.special.expit(X);

def standardize(X):
    '''
    @param training examples:
    @return standaridized samples:
    '''
    mean = np.mean(X, axis=0);
    std = np.std(X, axis=0)
    X -= mean
    X /= std;
    return X, mean.reshape((1, X.shape[1])), std.reshape((1, X.shape[1]));

def random_samples(X,Y):
    np.random.seed();
    indices = np.random.permutation(X.shape[0])
    X_new = copy.deepcopy(X);
    X_new = X_new[indices];
    Y_new = copy.deepcopy(Y);
    Y_new = Y_new[indices];
    return X_new,Y_new;
