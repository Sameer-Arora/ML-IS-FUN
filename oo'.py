## file to implement the neural networks.

import numpy as np;
import scipy;
from scipy.special import expit


def sigmoid(X):
    return scipy.special.expit(X);


def relu(X):
    return scipy.special.expit(X);


class Neuralnet():
    layer_dims = [];
    activa_layers = [];
    weights = [];
    bias = [];
    keep_probs = [];

    def __init__(self):

        pass

    def add_layer(self, li, input_dim=0, activation='sigmoid', keep_prob=1):

        if (len(self.layer_dims) == 0 and input_dim != 0):
            self.activa_layers.append(activation)
            self.layer_dims.append(input_dim);
            self.layer_dims.append(li);
            self.keep_probs.append(keep_prob);
            self.init_weights();

        elif (len(self.layer_dims) != 0 and input_dim != 0):
            print(" there is only one input layer!!");
        else:
            self.activa_layers.append(activation)
            self.layer_dims.append(li);
            self.keep_probs.append(keep_prob);
            self.init_weights();

    def init_weights(self):

        b = np.zeros((self.layer_dims[-1], 1))
        self.bias.append(b)
        # print("bias  " + str( b.shape ) )s

        w = np.random.rand(self.layer_dims[-2], self.layer_dims[-1]) * 0.02 + -0.01;

        # xavier's initalization
        # w = np.random.rand(self.layer_dims[-2], self.layer_dims[-1]) / np.sqrt(  self.layer_dims[-2] / 2.) ;

        self.weights.append(w)
        # print("weights  "+ str(w.shape) )
        ##print(w)

    def forward_pass(self, X, cache):

        a_prev = X;

        for i in range(len(self.layer_dims) - 1):
            W = self.weights[i];
            # print(W.shape)
            # print(a_prev.shape)
            a = np.matmul(a_prev, W);

            # Dropout training, notice the scaling of 1/p
            u1 = np.random.binomial(1, self.keep_probs[i], size=a.shape) / self.keep_probs[i];
            a = np.multiply(a, u1);

            cache["A"].append(a_prev);
            cache["u"].append(u1);

            if (self.activa_layers[i] == "sigmoid"):
                z = sigmoid(a);
            if (self.activa_layers[i] == "sigmoid"):
                z = sigmoid(a);
            elif (self.activa_layers[i] == "no"):
                z = a;
            # print("z")
            # print(z)

            cache["Z"].append(z);

            a_prev = a;

        return z;

    def compute_loss(self, Y, pred):
        if (self.activa_layers[-1] == "no"):
            return (np.linalg.norm(Y - pred) ** 2) / 2;

        elif (self.activa_layers[-1] == "sigmoid"):
            return - np.sum(np.matmul(Y.T, np.log(pred)) + np.matmul((1 - Y).T, np.log(1 - pred))) / Y.shape[0];

    def backward_pass(self, X, Y, Y_, cache):

        n = X.shape[0];
        grads_W = [];
        grads_b = [];

        a_prev = cache["A"][-1];
        u1 = cache["u"][-1];
        W = self.weights[-1];
        b = self.bias[-1];
        dz_prev = Y_ - Y;
        # print("a_prev")
        # print(a_prev)
        # print(np.matmul( a_prev .T ,Y-Y_  ) )

        if (self.activa_layers[-1] == "no"):
            diff = np.identity(n);

        if (self.activa_layers[-1] == "sigmoid"):
            diff = np.zeros((n, n));
            for i in range(n):
                # print( np.matmul(1 - Y_[i], Y_[i].T) ) ;
                diff[i][i] = np.matmul(1 - Y_[i], Y_[i].T);

            ##chk
            diff1 = np.diag(np.matmul(1 - Y_[i], Y_[i].T));
            print(diff - diff1)

        if (self.activa_layers[-1] == "relu"):
            diff = np.zeros((n, n));
            for i in range(n):
                # print( np.matmul(1 - Y_[i], Y_[i].T) ) ;
                if (Y_[i] <= 0):
                    diff[i][i] = 0;
                else:
                    diff[i][i] = 1;

            dig = np.zeros((1, Y_.shape[0]));
            dig[Y_ > 0] = 1;

            ##chk
            diff1 = np.diag(dig);
            print(diff - diff1)

        dw = np.matmul(a_prev.T, np.matmul(diff, dz_prev));
        dz = np.matmul(np.matmul(diff, dz_prev), W.T);
        db = np.matmul(dz_prev.T, np.matmul(diff, np.ones((n, 1))));

        # adding droput
        dz = np.multiply(dz, u1);
        # print( (Y_-Y).shape )
        # print(np.matmul(diff ,np.ones((n,1))).shape )
        # print(db.shape)
        # print(b.shape)
        assert (dw.shape == W.shape), "dw wrong!"
        assert (db.shape == b.shape), "db wrong!"
        # assert ( dz.shape == z.shape ),"dz wrong!"
        grads_b.append(db);
        grads_W.append(dw);
        cache["dZ"].append(dz);

        # print("dw")
        # print(dw)
        # print("dz")
        # print(dz)
        # print("db")
        # print(db)

        for i in range(len(self.layer_dims) - 3, -1, -1):
            # print(i)
            a_prev = cache["A"][i];
            z = cache["Z"][i];
            dz_prev = cache["dZ"][-1];

            W = self.weights[i];
            b = self.bias[i];
            # print("a_prev")
            # print(a_prev)
            # #print(np.matmul(a_prev.T, Y - Y_))

            if (self.activa_layers[-1] == "no"):
                diff = np.identity(n);
            if (self.activa_layers[-1] == "sigmoid"):
                diff = np.zeros((n, n));

                for i in range(n):
                    # print(np.matmul(1 - z[i], z[i].T));
                    diff[i][i] = np.matmul(1 - z[i], z[i].T);
                # print(diff)
                ##chk
                diff1 = np.diag(np.matmul(1 - Y_[i], Y_[i].T));
                print(diff - diff1)

            if (self.activa_layers[-1] == "relu"):
                diff = np.zeros((n, n));
                for i in range(n):
                    # print( np.matmul(1 - Y_[i], Y_[i].T) ) ;
                    if (Y_[i] <= 0):
                        diff[i][i] = 0;
                    else:
                        diff[i][i] = 1;

                dig = np.zeros((1, Y_.shape[0]));
                dig[Y_ > 0] = 1;

                ##chk
                diff1 = np.diag(dig);
                print(diff - diff1)

            # print(dz_prev.shape)
            # print(diff.shape)
            # print(W.shape)
            dw = np.matmul(a_prev.T, np.matmul(diff, dz_prev));
            dz = np.matmul(np.matmul(diff, dz_prev), W.T);
            db = np.matmul(dz_prev.T, np.matmul(diff, np.ones((n, 1))));

            assert (dw.shape == W.shape), "dw wrong!"
            assert (db.shape == b.shape), "db wrong!"
            # assert ( dz.shape == z.shape ),"dz wrong!"

            grads_b.append(db);
            grads_W.append(dw);
            cache["dZ"].append(dz);
            # print("dw")
            # print(dw)
            # print("dz")
            # print(dz)
            # print("db")
            # print(db)

        return grads_W, grads_b;

    def update_weights(self, grads_W, grads_b, n, learn_rate=0.001):

        for i in range(len(self.layer_dims) - 1):
            self.weights[i] = self.weights[i] - (learn_rate / n) * grads_W[-1 - i];
            self.bias[i] = self.bias[i] - (learn_rate / n) * grads_b[-1 - i];

    def train():
        pass
