#!/usr/bin/env python

#################################################################
# file name: ffnn_training_script.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 19-06-2022
# last edit: 19-06-2022 (Benoit)
# function: trains a feed-forward neural network on the MNIST data set and saves the final model
# inputs: MNIST data set and model parameters
# outputs: .npy file that contains the weight matrices and the parameters
#################################################################

from random import random
from ffnn_model import FFNN, CrossEntropy
import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST
import pickle

if __name__ == '__main__':

    # define model architecture
    input_dim = 28 * 28  # MNIST data set is 28 x 28 pixels each sample
    output_dim = 10  # MNIST are digits from 0 to 9
    num_layers = 2  # number of hidden layers
    hidden_layer_dim = 32  # number of neurons per hidden layer
    epochs = 1  # number of epochs
    batch_size = 8  # number of samples going through the model at each iteration during batch training
    step_size = 0.4 # Gradient descent step size alpha
    loss_function = CrossEntropy()  # instantiate loss function
    mnist_path = "../data/mnist-data"  # path to the mnist data folder
    load_weight_path = "../data/model_weights/weight_matrix_v1.pickle"  # path to the mnist data folder
    regularisation = 0.1

    # get train and test sets
    mnist = MNIST(mnist_path)
    mnist.gz = True
    trainX, trainY = mnist.load_training()
    testX, testY = mnist.load_testing()
    print("Train set size: " + str(len(trainX)))
    print("Test set size: " + str(len(testX)))
    assert len(trainX) == len(trainY)

    # Dataset -> np.array
    trainX = np.array(trainX)  # should be N x 784
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    assert trainX.shape[1] == 784

    # Standardize input
    mean = trainX.mean()
    std = trainX.std()
    trainX = (trainX - mean) / std
    testX = (testX - mean) / std

    # Batchs
    num_batches = int(np.floor(trainY.size / batch_size))
    BX = [trainX[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainX
    BY = [trainY[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainY

    
    with open(load_weight_path, 'rb') as handle:
        model_params =  pickle.load(handle)

    print(" model parameters :", model_params['b2'])

    # instantiate model
    model = FFNN(num_inputs=input_dim, num_outputs=output_dim, num_layers=num_layers, hidden_layer_dim=hidden_layer_dim, load_model=True, model_params=model_params)

    # Initialize stats arrays
    loss_test = np.zeros([epochs, 1])
    acc_test = np.zeros([epochs, 1])
    loss_train = np.zeros([epochs, 1])
    acc_train = np.zeros([epochs, 1])

    #Loop through data
    for epoch in range(epochs):
        print("epoch # :", epoch)


        for batch in range(int(len(BX)/500)):
            # Take the set from current batch
            X = BX[batch]
            Y = BY[batch]
            model.zero_grad()

            #Forward
            output = model.forward(X)

            # batch training loss and the derivative
            loss = loss_function.forward(output, Y)
            d_loss = loss_function.backward(output, Y)

            # backward propagation
            grads = model.backward(d_loss)

            # gradient descent step
            model.step(step_size=step_size)
        
        for key in model.model_params.keys():

            print("Key: ", key)
            print("Parameters: ", model.model_params[key][0:5])

            print("gradients: " , model.grads[key][0:5])

        # train and test loss of whole train set
        output = model.forward(trainX)
        y_pred = model.predict(trainX)
        loss_train[epoch] = loss_function.forward(output, trainY)
        acc_train[epoch] = np.mean(y_pred == trainY)

    print("Training completed")
    print("************")
    print("training accuracy :", acc_train)
    print("training loss :", loss_train)
    fig, ax = plt.subplots(1,2)


    ax[0].plot(list(range(epochs)),loss_train, color = 'r', label='loss_train')
    ax[0].legend()
    ax[1].plot(list(range(epochs)),acc_train, color = 'b', label='acc_train')
    ax[1].legend()

    plt.show()

"""
###


    for key in model.model_params:
        print("param shape : ", model.model_params[key].shape)

        # create batch of random inputs
    input = np.random.uniform(0.1, 0.6, (batch_size, input_dim))

    # create random ground truth 
    ground_truth = np.arange(batch_size)

    # perform forward pass on random input
    model_output = model.forward(input=input)

    print("model output: ", model_output)

    # compute loss
    loss = loss_function.forward(model_output, ground_truth)

    print("loss : ", loss)

    # compute backward pass on loss
    d_loss = loss_function.backward(model_output, ground_truth)

    print("d_loss : ", d_loss)

    # compute backward pass on model
    gradients = model.backward(d_loss)

    print("gradients : ", gradients)

####
"""
