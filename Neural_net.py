import copy
import csv
from matplotlib import pyplot as plt
import os
import pandas as pd
from model import Neuralnet
import pandas as pd
import cv2 as cv
import math
import numpy as np
from helper import standardize,random_samples
import plots

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
data = pd.read_csv('steering/data.txt', sep="\t", header=None)
# print(dataf_#)

# split into input (X) and output (Y) variableoutpuf_ts
I=[]
I = data.iloc[:, 0].values;
# print(I);

Y = np.zeros((data.shape[0], 1));
Y[:, 0] = data.iloc[:, -1];
training_f_X=I;
training_f_Y=Y;

# random sampling the training data.
indices = np.random.permutation(Y.shape[0])
training_idx, test_idx = indices[: round(0.80 * Y.shape[0])], indices[round(0.80 * Y.shape[0]):]
training_f_X, test_X = I[training_idx ], I[test_idx ]
training_f_Y, test_Y = Y[training_idx, :], Y[test_idx, :]

os.chdir("steering/");

## Experiment1
def Experiment1():

    mini_batch = 64;
    keep_prob = 1;
    # create model
    model = Neuralnet()
    model.add_layer (  512  ,1024 ,keep_prob=keep_prob,activation="sigmoid")
    model.add_layer (  64   ,keep_prob=keep_prob ,activation="sigmoid")
    model.add_layer (  Y.shape[1],keep_prob=keep_prob ,activation="no")

    # Compile model
    no_iterations=5000;
    learning_rate=0.01;
    print("Learning-Rate:- ", learning_rate," Batch_Size:- ", mini_batch)

    losses,te_losses=model.train(training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch,no_iterations)
    title="Exp1:- Batch size="+str(mini_batch)+" Learning Rate="+str(learning_rate)
    plots.linear_plot([x for x in range(len(losses))],losses,te_losses,"Iterations","Losses" ,title,1)
    plt.savefig(title + ".png")
    plt.show()

## Experiment2
def Experiment2():

    mini_batchs = [ 32,64,128 ];
    i=0;
    for mini_batch in mini_batchs:
        i+=1;
        keep_prob = 1;
        # create model
        model = Neuralnet()
        model.add_layer (  512  ,1024 ,keep_prob=keep_prob,activation="sigmoid")
        model.add_layer (  64   ,keep_prob=keep_prob ,activation="sigmoid")
        model.add_layer (  Y.shape[1],keep_prob=keep_prob ,activation="no")

        # Compile model
        no_iterations=1000;
        learning_rate=0.01;
        print("Learning-Rate:- ", learning_rate," Batch_Size:- ", mini_batch)

        losses,te_losses=model.train(training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch,no_iterations)
        print(losses)
        title = "Exp2:- Batch size=" + str(mini_batch) + " Learning Rate=" + str(learning_rate)
        plots.linear_plot([x for x in range(len(losses))],losses,te_losses,"Iterations","Losses", title ,1+i )
        plt.savefig(title + ".png")

        di = {"Iterations": [x for x in range(len(losses))], "Train Losses": losses, "Test Losses": te_losses};
        df = pd.DataFrame()
        df = df.append(di, ignore_index=True)

        df.to_csv("../2-"+i+ ".csv", sep='|')

    plt.show()


## Experiment3
def Experiment3():

    mini_batch = 64;
    keep_prob = 0.5;
    # create model
    model = Neuralnet()
    model.add_layer (  512  ,1024 ,keep_prob=keep_prob,activation="sigmoid")
    model.add_layer (  64   ,keep_prob=keep_prob ,activation="sigmoid")
    model.add_layer (  Y.shape[1],keep_prob=keep_prob ,activation="no")

    # Compile model
    no_iterations=1000;
    learning_rate=0.001;
    print("Learning-Rate:- ", learning_rate," Batch_Size:- ", mini_batch)

    losses,te_losses,out=model.train(training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch,no_iterations)
    title="Exp3:- Batch size="+str(mini_batch)+" Learning Rate="+str(learning_rate) +" with dropout"
    plots.linear_plot([x for x in range(len(losses))], losses ,te_losses,"Iterations","Losses", title,5 )
    plt.savefig(title + ".png")
    np.savetxt("../3-64.csv",np.array(out),delimiter="|" )
    plt.show()

## Experiment4
def Experiment4():
    i=0;
    learning_rates = [ 0.001,0.005 ];
    for learning_rate in learning_rates:
        mini_batch=64;
        keep_prob = 1;
        # create model
        model = Neuralnet()
        model.add_layer (  512  ,1024 ,keep_prob=keep_prob,activation="sigmoid")
        model.add_layer (  64   ,keep_prob=keep_prob ,activation="sigmoid")
        model.add_layer (  Y.shape[1],keep_prob=keep_prob ,activation="no")

        # Compile model
        no_iterations=1000;
        print("Learning-Rate:- ", learning_rate," Batch_Size:- ", mini_batch)
        losses,te_losses,out=model.train(training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch,no_iterations)
        title = "Exp4:- Batch size=" + str(mini_batch) + " Learning Rate=" + str(learning_rate)
        plots.linear_plot([x for x in range(len(losses))],losses,te_losses,"Iterations","Losses", title ,10+i)
        plt.savefig(title + ".png")
        np.savetxt("../4-"+ str(i) +".csv", np.array(out), delimiter="|")
        i+=1

    plt.show()

Experiment3()

# # pickl.
# with open('../results.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#
#     spamwriter.writerow(["with standardization"])
#     spamwriter.writerow([x for x in range(no_iterations) ])
#     spamwriter.writerow(losses)


## additionl work done

def RELU():

    mini_batch = 64;
    keep_prob = 1;
    # create model
    model = Neuralnet()
    model.add_layer (  512  ,1024 ,keep_prob=keep_prob,activation="relu")
    model.add_layer (  64   ,keep_prob=keep_prob ,activation="relu")
    model.add_layer (  Y.shape[1],keep_prob=keep_prob ,activation="no")

    # Compile model
    no_iterations=5000;
    learning_rate=0.01;
    print("Learning-Rate:- ", learning_rate," Batch_Size:- ", mini_batch)

    losses,te_losses=model.train(training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch,no_iterations)
    title="Relu:- Batch size="+str(mini_batch)+" Learning Rate="+str(learning_rate)
    plots.linear_plot([x for x in range(len(losses))],losses,te_losses,"Iterations","Losses" ,title,1)
    plt.savefig(title + ".png")
    plt.show()

def tanh():

    mini_batch = 64;
    keep_prob = 1;
    # create model
    model = Neuralnet()
    model.add_layer (  512  ,1024 ,keep_prob=keep_prob,activation="tanh")
    model.add_layer (  64   ,keep_prob=keep_prob ,activation="tanh")
    model.add_layer (  Y.shape[1],keep_prob=keep_prob ,activation="no")

    # Compile model
    no_iterations=5000;
    learning_rate=0.01;
    print("Learning-Rate:- ", learning_rate," Batch_Size:- ", mini_batch)

    losses,te_losses=model.train(training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch,no_iterations)
    title="tanh:- Batch size="+str(mini_batch)+" Learning Rate="+str(learning_rate)
    plots.linear_plot([x for x in range(len(losses))],losses,te_losses,"Iterations","Losses" ,title,1)
    plt.savefig(title + ".png")
    plt.show()

def adam():

    mini_batch = 64;
    keep_prob = 1;
    # create model
    model = Neuralnet()
    model.add_layer (  512  ,1024 ,keep_prob=keep_prob,activation="tanh")
    model.add_layer (  64   ,keep_prob=keep_prob ,activation="tanh")
    model.add_layer (  Y.shape[1],keep_prob=keep_prob ,activation="no")

    # Compile model
    no_iterations=5000;
    learning_rate=0.01;
    print("Learning-Rate:- ", learning_rate," Batch_Size:- ", mini_batch)

    losses,te_losses=model.train(training_f_X,training_f_Y,test_X,test_Y,learning_rate,mini_batch,no_iterations, )
    title="tanh:- Batch size="+str(mini_batch)+" Learning Rate="+str(learning_rate)
    plots.linear_plot([x for x in range(len(losses))],losses,te_losses,"Iterations","Losses" ,title,1)
    plt.savefig(title + ".png")
    plt.show()