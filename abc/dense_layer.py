import numpy as np
from activations import Activation

class Dense:

	def __init__(self, input_units, output_units, activation = None):
		"""
		Params:
		input_units = Number of input nodes
		output_units = Number of output nodes
		activation = The activation layer
		"""
		# self.weights = np.random.normal(0.0, 1.0/np.sqrt(input_units), (input_units, output_units))
		# self.bias = np.random.normal(0.0, 1.0/np.sqrt(input_units), (1, output_units))
		# self.weights = np.random.uniform(-0.01, 0.01, (input_units, output_units))
		self.weights = np.linspace(-0.01, 0.01, num = input_units*output_units)
		self.weights = self.weights.reshape((input_units, output_units))
		self.bias = np.zeros((1,output_units))
		self.activation = Activation(activation)

		# Initialize Other Things as Zero
		self.output_units = None
		self.grad_weights = 0
		self.grad_bias = 0


	def forward_pass(self, input_units):
		"""
		Params:
		input_units = the input nodes

		Returns:
		The output_units
		"""
		self.output_units = np.matmul(input_units, self.weights) + self.bias
		activated_output_units = self.activation.function(self.output_units)

		return activated_output_units


	def backward_pass(self, input_units, grad_activated_output):
		"""
		Params:
		input_units = input_units
		output_units = non-activated output units
		grad_activated = gradient of activated output units

		Returns:
		grad_activated_input
		"""
		grad_output_units = grad_activated_output*self.activation.derivative(self.output_units)
		self.grad_bias = grad_output_units.sum(axis=0)

		self.grad_weights = np.matmul(input_units.T, grad_output_units)
		grad_activated_input = np.matmul(grad_output_units, self.weights.T)

		return grad_activated_input

	def run(self, input_units):
		return self.forward_pass(input_units)


	def update(self, learning_rate):
		"""
		Params:
		learning_rate
		"""
		self.weights -= learning_rate*self.grad_weights
		self.bias    -= learning_rate*self.grad_bias


