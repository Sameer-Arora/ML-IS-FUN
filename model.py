## file to implement the neural networks.
import math
import numpy as np;
import scipy;

import copy;
from helper import standardize,random_samples
from scipy.special import expit
from scipy.misc import imread

def activate(X,func):
    # sigmoid activation
    if func=="sigmoid":
        X_=copy.deepcopy(X)
        return scipy.special.expit(X_);

    # relu activation
    if func == "relu":
        X_=copy.deepcopy(X)
        X_[X_<0]=0
        return  X_;

    # tanh activation
    if func == "tanh":
        X_=copy.deepcopy(X)
        X_=np.tanh(X_)
        return  X_;

    # no activation
    if func == "no":
        X_=copy.deepcopy(X)
        return  X_;


def derivative_activate(a,func):

    # sigmoid activation
    if (func == "sigmoid"):
        diff = np.multiply(1 - a, a);

    # tanh activation
    if (func== "tanh"):
        diff = 1 - np.multiply(a, a);

    # relu activation
    if (func== "relu"):
        diff = np.zeros((a.shape));
        diff[a > 0] = 1;

    # no activation
    if func == "no":
        diff = np.ones((a.shape));

    return diff

class Neuralnet():

    layer_dims = [];
    activa_layers = [];
    weights = [];
    bias = [];
    keep_probs = [];

    ## for adam
    m_t = []
    v_t = []
    iters = 0

    ## for bnorm
    gamma=[]
    beta=[]

    def __init__(self):
        self.layer_dims = [];
        self.activa_layers = [];
        self.weights = [];
        self.bias = [];
        self.keep_probs = [];
        self.m_t= [];
        self.v_t= [];
        self.gamma= [];
        self.beta= [];

    #function to dynamically add new layers to network.
    def add_layer(self, li, input_dim=0, keep_prob=1, activation='sigmoid',type=0):
        # print(len(self.layer_dims))
        if (len(self.layer_dims) == 0 and input_dim != 0):
            self.activa_layers.append(activation)
            self.layer_dims.append(input_dim);
            self.layer_dims.append(li);
            self.keep_probs.append(keep_prob);
            self.init_weights(type);

        elif ( len(self.layer_dims)!= 0 and input_dim != 0 ):

            print(" there is only one input layer!!");

        else:
            self.activa_layers.append(activation)
            self.layer_dims.append(li);
            self.keep_probs.append(keep_prob);
            self.init_weights(type);

    # function to initialze the network weights and biasses.
    def init_weights(self,type=0):

        np.random.seed();
        b = np.zeros((1,self.layer_dims[-1]))

        self.bias.append(b)
        if(type==0):
            w = np.linspace(-0.01, 0.01, num= self.layer_dims[-2] * self.layer_dims[-1])
            w = w.reshape((self.layer_dims[-2], self.layer_dims[-1]))
        else:
            # xavier's initalization of the weights
            w = np.linspace(-0.01, 0.01, num= self.layer_dims[-2] * self.layer_dims[-1])
            w = w.reshape((self.layer_dims[-2], self.layer_dims[-1])) / np.sqrt(  self.layer_dims[-2] / 2.) ;

        self.weights.append(w)
        self.m_t.append( np.zeros(w.shape) )
        self.v_t.append( np.zeros(w.shape) )

    # function for batch norm forward_pass
    def batchnorm_forward(X, gamma, beta):
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + 1e-8)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        return out, cache, mu, var

    # function for forward_pass to compute the L2 loss
    def forward_pass(self, X, cache,dropout,bnorm):

        a_prev = X;
        cache["A"].append(a_prev);

        for i in range(len(self.layer_dims) - 1):
            W = self.weights[i];
            b=self.bias[i];
            z = np.matmul(a_prev, W) + b ;

            if(dropout):
                # Dropout training, notice the scaling of 1/p
                u1 = np.random.binomial(1, self.keep_probs[i], size=z.shape) / self.keep_probs[i];
                # print(u1)
                z = np.multiply(z, u1);
                cache["u"].append(u1);

            if(bnorm):
                z, bn1_cache, mu, var = batchnorm_forward(z, self.gamma[i], self.beta[i]);

            a = activate(z,self.activa_layers[i]);
            # print(i, a.shape)
            a_prev = a;
            cache["A"].append(a_prev);

        return a_prev;

    # function for forward_pass to compute the L2 loss
    def predict(self, X):

        a_prev = X;

        for i in range(len(self.layer_dims) - 1):
            W = self.weights[i];
            b=self.bias[i];
            z = np.matmul(a_prev, W) + b ;
            a = activate(z,self.activa_layers[i]);
            a_prev = a;

        return a_prev;

    # function to compute the loss in network
    def compute_loss(self, Y, pred):

        if (self.activa_layers[-1] == "no"):
            return ( np.linalg.norm(Y - pred) ** 2 ) / 2;

        elif (self.activa_layers[-1] == "sigmoid"):
            return - np.sum(np.matmul(Y.T, np.log(pred)) + np.matmul((1 - Y).T, np.log(1 - pred))) / Y.shape[0];

    # function to compute the gradients flow in the network
    def backward_pass(self, X, Y, Y_, cache,dropout):

        grads_W = [];
        grads_b = [];

        a_prev = cache["A"][-2];
        da_prev = (Y_ - Y);

        diff=derivative_activate(Y_,self.activa_layers[-1]);
        dz = np.multiply(diff, da_prev);
        if(dropout):
            # adding droput
            u1 = cache["u"][-1];
            dz = np.multiply(dz, u1);

        dw = np.matmul(a_prev.T, dz);
        db = np.sum(dz,0).reshape(1,-1);

        grads_b.append(db);
        grads_W.append(dw);

        for i in range( len(cache["A"]) - 2 , 0, -1):
            a = cache["A"][i]
            a_prev = cache["A"][i-1]
            dz_prev = dz

            W = self.weights[i]
            diff = derivative_activate(a, self.activa_layers[i-1]);
            dz = np.multiply(diff, np.matmul(dz_prev,W.T));

            if (dropout):
                # adding droput
                u1 = cache["u"][i - 1];
                dz = np.multiply(dz, u1);

            dw = np.matmul(a_prev.T, dz);
            db = np.sum(dz,0).reshape(1,-1);

            grads_b.append(db);
            grads_W.append(dw);

        return grads_W, grads_b;


    def update_weights(self, grads_W, grads_b, learn_rate=0.1,opt=""):
        if(opt=="adam"):
            alpha = 0.01
            beta_1 = 0.9
            beta_2 = 0.999  # initialize the values of the parameters
            epsilon = 1e-8
            self.iters+=1

            for i in range(len(self.layer_dims) - 1):
                self.m_t[i] = beta_1 * self.m_t[i] + (1 - beta_1) *  grads_W[-1 - i] # updates the moving averages of the gradient
                self.v_t[i] = beta_2 * self.v_t[i] + (1 - beta_2) *( grads_W[-1 - i] *  grads_W[-1 - i] )  # updates the moving averages of the squared gradient
                m_cap = self.m_t[i] / (1 - (beta_1 ** self.iters))  # calculates the bias-corrected estimates
                v_cap = self.v_t[i] / (1 - (beta_2 ** self.iters ))  # calculates the bias-corrected estimates
                self.weights[i] = self.weights[i] - ( learn_rate) * m_cap /( np.sqrt(v_cap) + epsilon) ;
                self.bias[i] = self.bias[i] - (learn_rate ) * grads_b[-1 - i];

        else:
            for i in range(len(self.layer_dims) - 1):
                self.weights[i] = self.weights[i] - (learn_rate/64 ) * grads_W[-1 - i] ;
                self.bias[i] = self.bias[i] - (learn_rate//64 ) * grads_b[-1 - i];

    def train(self,training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch=64,no_iterations=1000,stad=True,dropout=False,opt=""):

        losses = [];
        te_losses = [];
        out=[];

        print("Start to train with:-" + opt)

        ## number of epochs
        for j in range(no_iterations):
            loss = 0;

            training_X,training_Y = random_samples(training_f_X,training_f_Y);

            for i in range( math.ceil(len(training_X) / mini_batch) ):
                # for i in range(1):
                if ((i + 1) * mini_batch - 1 < len(training_X)):
                    I_b = training_X[i * mini_batch: (i + 1) * mini_batch]
                    Y_b = training_Y[i * mini_batch:(i + 1) * mini_batch];
                    X_b = np.zeros((mini_batch, 1024));
                else:
                    I_b = training_X[i * mini_batch:]
                    Y_b = training_Y[i * mini_batch:];
                    X_b = np.zeros((Y_b.shape[0], 1024));

                ind = 0
                for inp_file in I_b:
                    X_b[ind, :] = imread(inp_file, flatten=True).flatten();
                    # normaliize the input.
                    if(stad):
                        X_b[ind, :] = ( X_b[ind, :] - min(X_b[ind, :] ) ) / ( max(X_b[ind, :]) -min(X_b[ind, :])  )  ;

                    ind += 1;
                if(stad):
                    X_b = (X_b - np.mean(X_b,0)) / (np.std(X_b,0) );

                cache = {"A": [], "Z": [], "dZ": [], "u": []}

                Y_ = self.forward_pass(X_b, cache,dropout);
                t_l = self.compute_loss(Y_b, Y_);
                loss += t_l;
                grads_W, grads_b = self.backward_pass(X_b, Y_b, Y_, cache,dropout);
                self.update_weights(grads_W, grads_b, learning_rate ,opt);

            loss /= training_X.shape[0];

            ind = 0;
            n=test_Y.shape[0];
            X_b = np.zeros((n, 1024));
            for inp_file in test_X:
                X_b[ind, :] = imread(inp_file, flatten=True).flatten();
                # standardize the input.
                if (stad):
                    X_b[ind, :] = (X_b[ind, :] - min(X_b[ind, :])) / (max(X_b[ind, :]) - min(X_b[ind, :]));

                ind += 1;
            if (stad):
                X_b = (X_b - np.mean(X_b, 0)) / (np.std(X_b, 0));

            Y_ = self.predict(X_b);
            te_loss = self.compute_loss(test_Y, Y_) / n;

            te_losses.append(te_loss);
            losses.append(loss);
            out.append( [ j,loss ,te_loss] );
            print("Iteration:-  " + str(j)+" | Train Loss:-  " + str(loss) + "| Test Loss:-  " + str(te_loss));

        return losses,te_losses,out;

    # def get_weights(self):
    #     re=[]
    #     for w in self.weights:
    #         re.append(w.flatten())
    #
    #     for w in self.bias:
    #         re.append(w.flatten())
    #
    #     return np.asarray(re);

    # def check_gradient(self, training_X, epsilon=1e-4):
    #
    #
    #     # assign the weight_vector as the network topology
    #     initial_weights = np.array(self.get_weights())
    #     numeric_gradient = np.zeros(initial_weights.shape)
    #     perturbed = np.zeros(initial_weights.shape)
    #     n_samples = float(training_X.shape[0])
    #
    #     print("[gradient check] Running gradient check...")
    #
    #     for i in range(len( self.weights ) ):
    #         perturbed[i] = epsilon
    #         right_side = self.error(initial_weights + perturbed, training_X )
    #         left_side = self.error(initial_weights - perturbed, training_X )
    #         numeric_gradient[i] = (right_side - left_side) / (2 * epsilon)
    #         perturbed[i] = 0
    #     # end loop
    #
    #     # Calculate the analytic gradient
    #     analytic_gradient = self.gradient(self.get_weights(), training_data, training_targets, cost_function)
    #
    #     # Compare the numeric and the analytic gradient
    #     ratio = np.linalg.norm(analytic_gradient - numeric_gradient) / np.linalg.norm(analytic_gradient + numeric_gradient)
    #
    #     if not ratio < 1e-6:
    #         print( "[gradient check] WARNING: The numeric gradient check failed! Analytical gradient differed by %g from the numerical." )\
    #
    #     return ratio
    # # end