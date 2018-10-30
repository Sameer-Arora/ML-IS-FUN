import numpy as np


class Flatten:

	def __init__(self):
		self.output_units = None
		pass

	def forward_pass(self, input_units):
		self.output_units = input_units.reshape(input_units.shape[0], input_units.shape[1]*input_units.shape[2]*input_units.shape[3])
		return self.output_units

	def backward_pass(self, input_units, grad_activated_output):
		return grad_activated_output.reshape(input_units.shape)

	def update(self, learning_rate):
		pass

	def run(self, input_units):
		return self.forward_pass(input_units)

# fl = Flatten()
# a = np.arange(96)
# a = a.reshape(2,3,4,4)
# b = fl.forward_pass(a)
# b = fl.backward_pass(a,b)
# print(b)