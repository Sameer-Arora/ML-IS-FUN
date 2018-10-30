import numpy as np

class Batchnorm():

    def __init__(self):

        self.gamma = None
        self.beta = None
        self.epsilon = 1e-8

    def forward_pass(self, input_units):
        """
        Params:
        input_units: input to the layer
        Returns:
        Normalized outputs with a different mean and variance according to gamma and beta

        First the input layer is flattened and mean and variance is caluclated
        Next the input is standardized by x' = (x - u)/s
        Lastly the distribution is shifted using gamma and beta: y = gamma*x' + beta

        """

        self.input_units_flattened = input_units.flatten().reshape(input_units.shape[0],-1)

        if self.gamma is None:
            self.gamma = np.ones((1, *(self.input_units_flattened.shape[1:])))
        if self.beta is None:
            self.beta = np.zeros((1, *(self.input_units_flattened.shape[1:])))

        self.mean = self.input_units_flattened.mean(axis = 0)
        self.variance = self.input_units_flattened.var(axis = 0)

        self.output_units = (self.input_units_flattened - self.mean)/np.sqrt(self.variance + self.epsilon)
        out = np.multiply(self.gamma, self.output_units) + self.beta

        return out.reshape(input_units.shape)

    def backward_pass(self,input_units, grad_activated_output):
        """

        """

        batch_size = input_units.shape[0]
        grad_activated_output = grad_activated_output.flatten().reshape(batch_size,-1)
        x_minus_mean = self.input_units_flattened - self.mean
        std_inv = 1.0/np.sqrt(self.variance + self.epsilon)
        self.grad_beta = grad_activated_output.sum(axis = 0)
        self.grad_gamma = np.multiply(grad_activated_output, self.output_units).sum(axis = 0)
        

        grad_input_units = np.multiply(self.gamma, std_inv)/(batch_size+self.epsilon) + \
                          (batch_size*grad_activated_output - np.multiply(self.grad_gamma, self.output_units) - self.grad_beta)

        return grad_input_units.reshape(input_units.shape)


    def run(self, input_units):
        return self.forward_pass(input_units)

    def update(self, learning_rate):

        self.gamma -= learning_rate*self.grad_gamma
        self.beta  -= learning_rate*self.grad_beta