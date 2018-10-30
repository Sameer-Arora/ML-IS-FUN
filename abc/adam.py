import numpy as np

class Adam:

	def __init__(self, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):

		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon

		self.m = 0
		self.v = 0
		self.t = 0

	def step_size(self, alpha, grad):
		"""
		Returns the new step size
		"""
		# Update timestep
		self.t += 1
		# Update first moment
		self.m = self.beta_1*self.m + (1-self.beta_1)*grad
		# Update second moment
		self.v = self.beta_2*self.v + (1-self.beta_2)*grad**2
		# Bias corrected first moment
		m_ = self.m/(1-np.power(self.beta_1, self.t))
		# Bias corrected second moment
		v_ = self.v/(1-np.power(self.beta_2, self.t))

		return alpha*m_/(np.sqrt(v_) + self.epsilon)
