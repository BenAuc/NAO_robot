#################################################################
# file name: ffnn_model.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 19-06-2022
# last edit: 19-06-2022 (Benoit)
# purpose: define the required class and methods to implement a feed-forward neural network solving a classification task
#################################################################

import numpy as np


class FFNN:

    def __init__(self, num_inputs, num_outputs, num_layers, hidden_layer_dim, load_model=False, model_params={}):
        """
        Intialization of the model
        """
        ### Initialize model parameters ###
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs 
        self.num_layers = num_layers
        self.hidden_layer_dim = hidden_layer_dim # number of neurons in each hidden layer
        self.activation_func = Sigmoid() # select activation function applied after each but the last later
        self.layer = Fully_Connected_Layer() # type of hidden layer is affine, fully-connected

        # class variable to save the gradients during the backward pass
        self.grads = {} 
        for i_layer in range(self.num_layers):
            self.grads['W' + str(i_layer + 1)] = 0.0

        # class variable to pass on to the backward pass some values computed during the forward pass
        self.cache = {}
        self.model_params = {}

        ### Initialize weight matrices and biases for all layers and store these in a dict ###
        # layer 1

        if not load_model:
            print("initializing model with random parameters")
            self.model_params = {'W1': np.random.normal(-0.025, 0.025, (self.num_inputs, self.hidden_layer_dim)), 
                                'b1': np.zeros(self.hidden_layer_dim)}

            # subsequent hidden layers
            for i_layer in range(self.num_layers - 2):
                self.model_params['W' + str(i_layer + 2)] = np.random.normal(-0.025, 0.025, (self.hidden_layer_dim, self.hidden_layer_dim))
                self.model_params['b' + str(i_layer + 2)] = np.zeros(self.hidden_layer_dim)

            # output layer
            self.model_params['W' + str(self.num_layers)] = np.random.normal(-0.025, 0.025, (self.hidden_layer_dim, self.num_outputs))
            self.model_params['b' + str(self.num_layers)] = np.zeros(self.num_outputs)

        else:

            print("loading model parameters")
            self.model_params = model_params

    def forward(self, input):
        """
        Compute forward pass of the model

        Inputs:
        -input: input to the model of shape (N, D) where N is the # of samples in the batch and D the # of features in each sample

        Outputs:
        -output: predicted category for the given input, ranging from 0 to num_outputs in the model
        """
        # flatten the input
        X = input.reshape(input.shape[0], -1)

        # reset to 0 the dictionary to store cached values during forward pass
        self.cache = {}

        # perform forward pass on each layer
        for i_layer in range(self.num_layers):

            # print("forward pass on layer # :", i_layer)

            # fetch the layer parameters
            W, b = self.model_params['W' + str(i_layer + 1)], self.model_params['b' + str(i_layer + 1)]

            # perform forward pass
            X, cache_layer = self.layer.forward(x=X, w=W, b=b)
            self.cache['L' + str(i_layer + 1)] = cache_layer

            # perform activation on all but the last layer
            if i_layer < self.num_layers - 1:
                # print("activation function")
                X, cache_activation = self.activation_func.forward(input=X)
                self.cache['activation' + str(i_layer + 1)] = cache_activation

        output = X

        return output


    def backward(self, d_loss):
        """
        Compute forward pass of the model

        Inputs:
        -d_loss: upstream gradient i.e. gradient of loss function

        Outputs:
        -output: gradients of the model output with respect to each weight and bias
        """

        # fetch values cached during forward pass
        cache_layer = self.cache['L' + str(self.num_layers)]

        # perform backward pass on last layer of the network
        dh, dW, db = self.layer.backward(cache = cache_layer, d_upstream=d_loss)

        # store gradients
        self.grads['W' + str(self.num_layers)] = dW
        self.grads['b' + str(self.num_layers)] = np.mean(db,axis=0)

        # perform backward pass on subsequent layers
        for i_layer in range(self.num_layers - 2, -1, -1):

            # fetch values cached during forward pass
            cache_activation = self.cache['activation' + str(i_layer + 1)]
            cache_layer = self.cache['L' + str(i_layer + 1)]

            # backward pass on activation
            dh = self.activation_func.backward(cache = cache_activation, d_upstream = dh)

            # backward pass on fully-connected layer
            dh, dW, db = self.layer.backward(cache = cache_layer, d_upstream=dh)

            # store gradients
            self.grads['W' + str(i_layer + 1)] = dW
            self.grads['b' + str(i_layer + 1)] = np.mean(db,axis=0)

        return self.grads

    def predict(self, input):
        """
        Predicts output of the model
        Input:
        -input: input to the model of shape (N, D) where N is the # of samples in the batch and D the # of features in each sample

        Output:
        -predictions: numpy array of shape (N, 1) for predicted class

        """

        output = self.forward(input) # output is N x 10

        # logits are first normalized between 0 and 1 and then we take the exponential
        output_exp = np.exp(output - np.max(output, axis=1, keepdims=True))

        # softmax function: each exponential is divided by the sum of all scores
        output_probs = output_exp / np.sum(output_exp, axis=1, keepdims=True)

        predictions = np.argmax(output_probs, axis=1)

        return predictions

    def step(self, step_size):
        """
        Update weights
        Inputs:
        - step_size: Step size of gradient descent
        """
        # Update weights with gradient descent
        for key in self.model_params.keys():
            # print("Key: ", key)
            # print("Parameters: " , self.model_params[key][0:5])

            self.model_params[key] = self.model_params[key] - step_size * self.grads[key]
            
            # print("Parameters: " , self.model_params[key][0:5])

            # print("gradients: " , self.grads[key][0:5])


    def zero_grad(self):
        """
        Resets the gradients to zero
        """
        for key in self.grads.keys():
            self.grads[key] = self.grads[key]*0.0


    def save_model(self, path):
        """
        Saves the model parameters
        """

        np.save(path, self.model_params)






class Fully_Connected_Layer:

    def __init__(self):
        pass

    def forward(self, x, w, b):
        """
        Compute forward pass of a fully-connected layer (y = w * x + b)

        Inputs:
        -x: input to the layer of shape (N, D) where N is the # of samples in the batch and and D the number of features in each sample
        -w: array of weights of shape (D, M) where M is the number of neurons in the layer
        -b: array of biases of shape (M, 1)

        Outputs:
        -output: result when the layer is applied to the input, array of shape (N, M)
        -cache: values necessary to compute the backward pass, tuple (x, w, b)
        """
        # print("shape input :", x.shape)
        # print("weight input :", w.shape)
        # extract # of samples in batch
        N = x.shape[0]

        #compute output of layer
        output = np.matmul(x.reshape(N, -1), w) + b

        cache = (x, w, b)

        return output, cache

    def backward(self, cache, d_upstream):
        """
        Compute forward pass of a fully-connected layer (y = w * x + b)

        Inputs:
        -cache: tuple (x, w, b) containing the input to the layer (x) of shape (N, D), the layer weight matrix (w) of shape (D, M), the layer biases (b) of shape (M, 1)
        -d_upstream: upstream gradient of shape (M, 1) where M is the number of neurons in the layer

        Outputs:
        -dx: gradient of the output of the layer with respect to its input multiplied by the upstream gradient (chain rule), shape (N, D)
        -dw: gradient of the output of the layer with respect to its weights multiplied by the upstream gradient (chain rule), shape (D, M)
        -db: gradient of the output of the layer with respect to its biases multiplied by the upstream gradient (chain rule), shape (M, 1)
        """
        # unpack the values cached during the forward pass
        x, w, b = cache

        # extract # of samples in batch
        N = x.shape[0]

        # compute gradient with respect to input: d_layer_output / dx = w
        # multiply this with the upstream gradient
        dx = np.matmul(d_upstream, w.T)

        # compute gradient with respect to weight: d_layer_output / dw = x
        # multiply this with the upstream gradient
        dw = np.matmul(x.T, d_upstream) / N

        # compute gradient with respect to biases: d_layer_output / db = 1
        # multiply this with the upstream gradient
        db = d_upstream / N

        return dx, dw, db

class Sigmoid:

    def forward(self, input):
        """
        Compute output of the sigmoid function

        Inputs:
        -input: input to the sigmoid function

        Outputs:
        -output: result when the sigmoid is applied to the input
        -cache: values necessary to compute the backward pass
        """
        output = np.divide(1, 1 + np.exp(-input))
        
        cache = output

        return output, cache

    def backward(self, cache, d_upstream):
        """
        Compute backward pass of the sigmoid function.

        Inputs:
        -cache: result of the forward pass of the sigmoid function
        -d_upstream: upstream gradient

        Outputs:
        -dx: gradient of the sigmoid multiplied by upstream derivative
        """
        
        dx = d_upstream * cache * (1-cache)

        return dx


class Relu:

    def forward(self, input):
        """
        Compute output of the Relu function

        Inputs:
        -input: input to the function

        Outputs:
        -output: result when the relu is applied to the input
        -cache: values necessary to compute the backward pass
        """
        output = np.maximum(0, input)
        
        cache = input

        return output, cache

    def backward(self, cache, d_upstream):
        """
        Compute derivative of the Relu function.

        Inputs:
        -cache: result of the forward pass 
        -d_upstream: upstream gradient

        Outputs:
        -dx: gradient of relu multiplied by upstream derivative
        """
        
        dx = d_upstream * (input > 0) * 1

        return dx


class CrossEntropy:

    def __init__(self):
        # class variable to pass on to the backward pass some values computed during the forward pass
        self.cache = {}

    def forward(self, model_output, ground_truth):
        """
        Compute forward pass i.e. compute cross entropy loss

        Inputs:
        -model_output: model prediction, array of dim (N, C) where N is # samples in batch and C is # of classes
        -ground_truth: expected output, array of dim (N, 1) containing the class corresponding to each sample

        Outputs:
        -loss: the cross-entropy loss value
        """
        # extract number of samples and dimension of each sample in the batch
        N, D = model_output.shape
        # print("N, D :", N, D)

        # transform ground truth labels into one hot encodings
        # for each sample the column corresponding to the ground truth class is set to one
        ground_truth_encoded = np.zeros_like(model_output)
        ground_truth_encoded[np.arange(N), ground_truth] = 1
        # print("encodings : ", ground_truth_encoded)

        # transform the logits the model outputted into a softmax distribution
        # logits are first normalized between 0 and 1 and then we take the exponential
        model_output_exp = np.exp(model_output - np.max(model_output, axis=1, keepdims=True))
        # print("model_output_exp : ", model_output_exp)

        # softmax function: each exponential is divided by the sum of all scores
        model_output_probs = model_output_exp / np.sum(model_output_exp, axis=1, keepdims=True)
        # print("model_output_probs : ", model_output_probs)

        # compute the loss for each sample
        loss = -ground_truth_encoded * np.log(model_output_probs)
        # print("loss : ", loss)
        loss = loss.sum(axis=1).mean()
        
        # cache computed value for the backward pass
        self.cache['probs'] = model_output_probs

        return loss

    def backward(self, model_output, ground_truth):
        """
        Compute backward pass i.e. derivative of the cross entropy loss

        Inputs:
        -model_output: model prediction, array of dim (N, C) where N is # samples in batch and C is # of classes
        -ground_truth: expected output, array of dim (N, 1) containing the class corresponding to each sample

        Outputs:
        -d_loss: array of cross entropy loss gradients 
        """
        # extract number of samples and dimension of each sample in the batch
        N, D = model_output.shape

        # compute derivative of the cross entropy with respect to each class and each sample in the batch
        d_loss = self.cache['probs']
        d_loss[np.arange(N), ground_truth] -= 1
        d_loss /= N

        return d_loss


class MSE:

    def __init__(self):
        # class variable to pass on to the backward pass some values computed during the forward pass
        self.cache = {}

    def forward(self, model_output, ground_truth):
        """
        Compute forward pass i.e. compute mean squared error

        Inputs:
        -model_output: model prediction, array of dim (N, D) where N is # samples in batch and D features
        -ground_truth: expected output, array of dim (N, D) containing the values corresponding to each sample

        Outputs:
        -loss: the mean squared error
        """
        # extract number of samples and dimension of each sample in the batch
        N, D = model_output.shape

        # compute mean squared error
        loss = np.mean(0.5 * (ground_truth - model_output) **2)

        return loss

    def backward(self, model_output, ground_truth):
        """
        Compute backward pass i.e. derivative of the mean squared error

        Inputs:
        -model_output: model prediction, array of dim (N, D) where N is # samples in batch and D features
        -ground_truth: expected output, array of dim (N, D) containing the values corresponding to each sample

        Outputs:
        -d_loss: array of mean squared error gradients of dimensions (1,D)
        """
        # extract number of samples and dimension of each sample in the batch
        N, D = model_output.shape

        # compute derivative of the cross entropy with respect to each class and each sample in the batch
        d_loss = - ground_truth + model_output

        return d_loss




