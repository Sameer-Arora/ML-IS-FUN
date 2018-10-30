import numpy as np
from model import Model
from dense_layer import Dense
from dropout import Dropout
from convnet import Conv2D
from maxpool import Maxpool
from flatten import Flatten
from batchnorm import Batchnorm
from adam import Adam
import pandas as pd
import utils

#from keras.models import Sequential
#import keras.layers as L
#from keras.optimizers import SGD
def getModel(X, Y, val_X, val_Y):
	model = Model(learning_rate = 0.01, batch_size = 64, epochs = 1000, optimizer = None)
	# model.add(Dropout(0.5))
	model.add(Dense(1024, 512, activation = 'sigmoid'))
	# model.add(Batchnorm())
	# model.add(Dropout(0.5))
	model.add(Dense(512, 64, activation = 'sigmoid'))
	# model.add(Batchnorm())
	# model.add(Dropout(0.5))
	model.add(Dense(64, 1))
	print("starting to train")
	model.train(X,Y, val_X, val_Y)
	model.save_history("4-05.csv")

	# model.add(Conv2D(16,(3,3), activation = 'relu'))
	# model.add(Maxpool((3,3), stride = 2)) # 16x16
	# model.add(Batchnorm())
	# model.add(Dropout(0.5))
	# model.add(Conv2D(4,(3,3), activation = 'relu'))
	# model.add(Maxpool((3,3), stride = 2)) # 8x8
	# model.add(Batchnorm())
	# model.add(Dropout(0.5))
	# model.add(Conv2D(1,(3,3), activation = 'relu'))
	# model.add(Maxpool((3,3), stride = 2)) # 8x8
	# model.add(Batchnorm())
	# model.add(Dropout(0.5))
	# model.add(Flatten())
	# model.add(Dense(16,1))
	# model.train(X,Y)


	# model.add(Dense(20,10, activation = 'relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(10,1))
	# model.train(X,Y)
		

	#model.train(X, Y, lr, epochs)
	return model

def prepareTrainData(X, Y):
	

	for i in range(X.shape[0]):
		X[i,:] = X[i,:] -X[i,:].min()
		X[i,:] = X[i,:]/X[i,:].max() if X[i,:].max()>0 else 0

	means = []
	stds = []
	for i in range(X.shape[1]):
		means.append(X[:,i].mean())
		stds.append(X[:,i].std())
		X[:, i] = (X[:, i]-X[:, i].mean())/X[:, i].std()

	return X, Y, means, stds

def prepareTestData(X,Y,means,stds):
	
	for i in range(X.shape[0]):
		X[i,:] = X[i,:] -X[i,:].min()
		X[i,:] = X[i,:]/X[i,:].max() if X[i,:].max()>0 else 0


	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - means[i])/stds[i]

	return X,Y

X, Y = utils.get_data('steering', 0)
# print(X.shape)
X_train, Y_train, means, stds = prepareTrainData(X[:17600],Y[:17600])
X_test, Y_test = X[17600:], Y[17600:]
X_val, Y_val = prepareTestData(X_test, Y_test, means, stds)
model = getModel(X_train, Y_train, X_val, Y_val)
