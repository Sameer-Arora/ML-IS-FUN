import numpy as np
from im2col import *

class Maxpool:

	def __init__(self, kernel_size, stride = 1, padding ='same'):

		self.kernel_size = kernel_size
		self.stride = stride
		self.output_units = None
		self.padding = (kernel_size[0]-1)//2 if padding == 'same' else 0

	def forward_pass(self, input_units):
		"""
		Params:
		input_units = the input nodes

		Returns:
		The output_units
		"""
		batch_size, channels, in_height, in_width = input_units.shape

		out_height = (in_height - self.kernel_size[0] + 2*self.padding) // self.stride + 1
		out_width = (in_width - self.kernel_size[1] + 2*self.padding) // self.stride + 1

		x_split = input_units.reshape(batch_size * channels, 1, in_height, in_width)
		self.x_cols = im2col_indices(x_split, self.kernel_size[0], self.kernel_size[1], self.padding, self.stride)
		self.x_cols_argmax = np.argmax(self.x_cols, axis=0)

		x_cols_max = self.x_cols[self.x_cols_argmax, np.arange(self.x_cols.shape[1])]
		out = x_cols_max.reshape(out_height, out_width, batch_size, channels).transpose(2, 3, 0, 1)

		return out



	def backward_pass(self, input_units, grad_activated_output):
		"""
		"""
		batch_size, channels, in_height, in_width = input_units.shape

		grad_output = grad_activated_output.transpose(2, 3, 0, 1).flatten()
		grad_x_col = np.zeros_like(self.x_cols)
		grad_x_col[self.x_cols_argmax, np.arange(grad_x_col.shape[1])] = grad_output
		grad_activated_inputs = col2im_indices(grad_x_col, (batch_size * channels, 1, in_height, in_width), self.kernel_size[0], self.kernel_size[1], self.padding, self.stride)
		grad_activated_inputs = grad_activated_inputs.reshape(input_units.shape)

		return grad_activated_inputs

	def run(self, input_units):
		return self.forward_pass(input_units)


	def update(self, learning_rate):
		pass