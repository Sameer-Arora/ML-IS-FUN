import numpy as np
import math
import time


class BatchNormalization(object):
    def __init__(self, input_shape, momentum=0.9):
        '''
        # params :
        input_shape :
        momentum : momentum for exponential average
        '''
        self.input_shape = input_shape
        self.momentum = momentum
        self.insize = input_shape[1]

        # random setting of gamma and beta, setting initial mean and std
        rng = np.random.RandomState(int(time.time()))
        self.gamma = np.asarray(rng.uniform(low=-1.0 / math.sqrt(self.insize), high=1.0 / math.sqrt(self.insize), size=(input_shape[1])))
        self.beta = np.zeros((input_shape[1]))
        self.mean = np.zeros((input_shape[1]))
        self.var = np.ones((input_shape[1]))

        # parameter save for update
        self.params = [self.gamma, self.beta]

    def get_result(self, input):
        # returns BN result for given input.
        epsilon = 1e-06

        now_mean = np.mean(input, axis=0)
        now_var = np.var(input, axis=0)
        now_normalize = (input - now_mean) / np.sqrt(now_var + epsilon)  # should be broadcastable..
        output = self.gamma * now_normalize + self.beta
        # mean, var update
        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * now_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * (
                    self.input_shape[0] / (self.input_shape[0] - 1) * now_var)
        return output

    # changing shape for CNN mode
    def change_shape(self, vec):
        return T.repeat(vec, self.input_shape[2] * self.input_shape[3]).reshape(
            (self.input_shape[1], self.input_shape[2], self.input_shape[3]))

