#!/usr/bin/env python

#################################################################
# file name: ffnn_classification_MNIST_training_script.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 21-06-2022
# last edit: 23-06-2022 (Benoit, Vildana)
# function: trains a feed-forward neural network on the MNIST data set and saves the final model
# inputs: MNIST data set
# outputs: pickle file that contains the weight matrices, training statistics and the loss/accuracy curves
#################################################################

from ffnn_model import FFNN, CrossEntropy
import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST
import pickle

if __name__ == '__main__':

    # define model architecture
    input_dim = 28 * 28  # MNIST data set is 28 x 28 pixels each sample
    output_dim = 10  # MNIST are digits from 0 to 9
    num_layers = 4  # number of all layers (excluding the inputs)
    hidden_layer_dim = 32  # number of neurons per hidden layer
    epochs = 100  # number of epochs
    batch_size = 8  # number of samples going through the model at each iteration during batch training
    step_size = 0.05 # Gradient descent sstep size alpha
    loss_function = CrossEntropy()  # instantiate loss function
    mnist_path = "../data/mnist-data"  # path to the mnist data folder
    save_weight_path = "../data/model_weights_MNIST/weight_matrix_final.pickle"  # path to the mnist data folder
    save_statistics_path = "../data/model_weights_MNIST/fnn_statistics_final.pickle" 
    regularisation = 0.1

    # get train and test sets
    mnist = MNIST(mnist_path)
    mnist.gz = True
    trainX, trainY = mnist.load_training()
    testX, testY = mnist.load_testing()

    # Dataset -> np.array
    trainX = np.array(trainX)  # should be N x 784
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    # Standardize input
    mean = trainX.mean()
    std = trainX.std()
    trainX = (trainX - mean) / std
    testX = (testX - mean) / std

    # Batchs
    num_batches = int(np.floor(trainY.size / batch_size))
    BX = [trainX[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainX
    BY = [trainY[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainY

    # instantiate model
    model = FFNN(num_inputs=input_dim, num_outputs=output_dim, num_layers=num_layers, hidden_layer_dim=hidden_layer_dim, load_model=False, model_params={})

    # Initialize stats arrays
    loss_test = np.zeros([epochs, 1])
    acc_test = np.zeros([epochs, 1])
    loss_train = np.zeros([epochs, 1])
    acc_train = np.zeros([epochs, 1])

    #Loop through data
    for epoch in range(epochs):
        print("epoch # :", epoch)


        for batch in range(int(len(BX))):
            # Take the set from current batch
            X = BX[batch]
            Y = BY[batch]
            model.zero_grad()

            #Forward
            output = model.forward(X)

            # batch training loss and the derivative
            loss = loss_function.forward(output, Y)
            d_loss = loss_function.backward(output, Y)
            if batch % 1000 == 0:
                print("loss for batch # " + str(batch) + " is : ", loss)

            # backward propagation
            grads = model.backward(d_loss)

            # gradient descent step
            model.step(step_size=step_size)

        # train and test loss of whole train set
        output = model.forward(trainX)
        y_pred = model.predict(trainX)

        loss_train[epoch] = loss_function.forward(output, trainY)
        acc_train[epoch] = np.mean(y_pred == trainY)

        # train and test loss of whole test set

        output_test = model.forward(testX)
        y_pred_test = model.predict(testX)

        loss_test[epoch] = loss_function.forward(output_test, testY)
        acc_test[epoch] = np.mean(y_pred_test == testY)

    with open(save_weight_path, 'wb') as handle:
        pickle.dump(model.model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    acc_loss_final = {'loss_train':loss_train, 'acc_train':acc_train, 'loss_test':loss_test, 'acc_test':acc_test}
    
    with open(save_statistics_path, 'wb') as handle:
        pickle.dump(acc_loss_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed")
    print("************")
    print("training accuracy :", acc_train)
    print("training loss :", loss_train)

    font_size = 18
    fig, ax = plt.subplots(1,2)

    ax[0].set_title("Loss over epochs for training and test datasets", fontsize=font_size)
    ax[0].plot(list(range(epochs)),loss_train, color = 'r', label='loss_train', linewidth=2)
    ax[0].plot(list(range(epochs)),loss_test, color = 'b', label='loss_test')
    ax[0].set_ylabel("Cross-entropy loss", fontsize=font_size)
    ax[0].set_xlabel("Epoch", fontsize=font_size)
    ax[0].legend()

    ax[1].set_title("Prediction accuracy over epochs for training and test datasets", fontsize=font_size)
    ax[1].plot(list(range(epochs)),acc_train, color = 'r', label='acc_train ' + str(100*acc_train[-1][0]) + ' %', linewidth=2)
    ax[1].plot(list(range(epochs)),acc_test, color = 'b', label='acc_test ' + str(100*acc_test[-1][0]) + ' %')
    ax[1].set_ylabel("Accuracy (%)", fontsize=font_size)
    ax[1].set_xlabel("Epoch", fontsize=font_size)
    ax[1].legend(loc='lower right')

    plt.show()