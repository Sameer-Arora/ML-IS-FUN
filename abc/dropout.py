import numpy as np


class Dropout:

	def __init__(self, keep_prob):

		self.keep_prob = keep_prob
		self.mask = None
		self.output_units = None

	def forward_pass(self, input_units):
		"""
		Params:
		input_units = the input nodes

		Returns:
		The output_units
		"""

		self.mask = np.random.binomial(1,self.keep_prob, size=input_units.shape)/self.keep_prob
		self.output_units = input_units*self.mask
		return self.output_units


	def backward_pass(self, input_units, grad_activated_output):
		"""
		"""
		return grad_activated_output*self.mask

	def run(self, input_units):
		return input_units


	def update(self, learning_rate):
		pass