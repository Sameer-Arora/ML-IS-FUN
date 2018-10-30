import numpy as np
from scipy.special import expit as sigmoid

class Activation:

	def __init__(self, activation):

		self.function = lambda x: x
		self.derivative = lambda x : 1

		if activation is not None:

			if activation == "sigmoid":
				self.function = lambda x : sigmoid(x)
				self.derivative = lambda x : sigmoid(x)*(1-sigmoid(x))

			if activation == "tanh":
				self.function = lambda x : np.tanh(x)
				self.derivative = lambda x : 1 - np.square(np.tanh(x))

			if activation == "relu":
				self.function = lambda x: (x>0)*x
				self.derivative = lambda x: (x>0)*1

			if activation == "leaky_relu":
				self.function = lambda x: (x>0)*x + (x<0)*0.01*x
				self.derivative: lambda x: (x>0)*1 + (x<0)*0.01
