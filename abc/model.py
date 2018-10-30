import numpy as np

class Model:

	def __init__(self, learning_rate = 0.001, batch_size = 1, epochs = 100, optimizer = None):
		"""
		Params:
		learning_rate = Learning Rate of the Model
		"""
		self.layers = []
		self.units = [None]
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.optimizer = None
		self.history = np.zeros((epochs,3))


	def generate_batches(self, X, Y):

		# print(X.shape, Y.shape)
		indices = np.arange(X.shape[0])
		# np.random.shuffle(indices)
		# concat = np.concatenate((X,Y), axis = 1)
		# X,Y = concat[:,:-1], concat[:,-1]
		X, Y = X[indices], Y[indices]
		batch_count = X.shape[0]//self.batch_size + 1
		for i in range(batch_count):
			start = i*self.batch_size
			end = (i+1)*self.batch_size if ((i+1)< batch_count) else X.shape[0]
			yield X[start:end], Y[start:end]



	def train(self, X, Y, val_X, val_Y):
		
		for epoch in range(self.epochs):

			# loss = 0
			for x,y in self.generate_batches(X,Y):

				if(x.shape[0] == 0): continue

				self.units[0] = x
				for i, layer in enumerate(self.layers):
					self.units[i+1] = layer.forward_pass(self.units[i])

				error = (self.units[-1]-y)
				# print(error.sum())
				# loss += np.square(error).sum()/2

				learning_rate = self.learning_rate if self.optimizer is None else self.optimizer.step_size(self.learning_rate, error.sum())

				for layer, unit in zip(reversed(self.layers), reversed(self.units[:-1])):
					error = layer.backward_pass(unit,error)
					layer.update(learning_rate)
				
			# Get Training Loss
			train_loss = 0
			unit = X
			for i,layer in enumerate(self.layers):
				unit = layer.run(unit)
			train_loss = np.square(unit-Y).mean()/2

			# Get Validation Loss
			val_loss = 0
			unit = val_X
			for i,layer in enumerate(self.layers):
				unit = layer.run(unit)
			val_loss = np.square(unit-val_Y).mean()/2

			self.history[epoch,0] = epoch
			self.history[epoch,1] = train_loss
			self.history[epoch,2] = val_loss
				
			print("Epoch: {:>4} | Training: {:.10f} | Validation: {:.10f} ".format\
				(self.history[epoch,0]+1, self.history[epoch,1], self.history[epoch,2]))

	def predict(self, X):
		unit = X
		for i,layer in enumerate(self.layers):
			unit = layer.run(unit)
		return unit

	def save_history(self, filename):
		np.savetxt(filename, self.history, delimiter=',')

	def add(self, layer):
		self.layers.append(layer)
		self.units.append(None)

