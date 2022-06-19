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
import rospy
import numpy as np
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def input_normalization(coordinates):
    #####
    # this method takes as input a python list 1x2 corresponding to (x,y) coordinates and normalizes it
    # Inputs:
    #  coordinates: array of dim 2 x 1 containing the (x, y) pixel coordinates
    # Outputs:
    #  array of dim 2 x 1 containing the (x, y) pixel coordinates normalized to the iamge resolution
    #####
    # define camera resolution
    # pixels idx run from 0 to resolution - 1

    cam_y_max = 240 - 1
    cam_x_max = 320 - 1
    return [float(coordinates[0]) / cam_x_max, float(coordinates[1]) / cam_y_max]

def output_normalization(targets):
    # Normalize the targets in the training data

    pitch = targets[:, 0]
    roll = targets[:, 1]
    max_pitch = 0.06
    min_pitch = -0.63
    max_roll = 0.39
    min_roll = -0.32
    
    norm_pitch = np.array((pitch - min_pitch) / (max_pitch - min_pitch))
    norm_roll = np.array((roll - min_roll) / (max_roll - min_roll))
    norm_targets = np.column_stack((norm_pitch, norm_roll))

    return norm_targets

def output_denormalization(norm_targets):
    pitch = norm_targets[0]
    roll = norm_targets[1]
    max_pitch = 0.06
    min_pitch = -0.63
    max_roll = 0.39
    min_roll = -0.32

    pitch = np.array(pitch * (max_pitch - min_pitch) + min_pitch)
    roll = np.array(roll * (max_roll - min_roll) + min_roll)
    targets = [pitch, roll]

    return targets


def train_ffnn(cmac_weight_table, data, num_epochs, plot_table):
    # Train the CMAC
    # Inputs:
    #   cmac_weight_table: untrained weight table
    #   data: training data of dim N samples x 4 where each sample is a (x, y) pixel coordinates 
    #       followed by (shoulder_pitch, shoulder_roll) joint state
    #   num_epochs: number of epochs for the training
    #   plot_table: flag to plot the weight table after each epoch
    # Outputs:
    #   new_cmac_weight_table: trained weight table
    # Example call:
    #   cmac_weight_table = train_cmac(cmac_weight_table, training_dataset[0:149,:], 10)

    # Initialize variables
    new_cmac_weight_table = np.zeros(cmac_weight_table.shape) # Trained weight table
    alpha = 0.02 # Learning rate
    inputs = data[:, 0:2] # Inputs
    t = data[:,-2:] # Targets (ground truth)
    MSE_sample = np.zeros((data.shape[0], num_epochs)) # MSE of each sample at every epoch
    MSE = [0] * num_epochs # General MSE at every epoch

    # Target training:
    print('******')
    print("Starting target training...")
    
    # Repeat num_epochs times
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + 1) + "/" + str(num_epochs))

        np.random.shuffle(np.asarray(data))

        inputs = data[:, 0:2]   # Inputs
        t = data[:,-2:]         # Targets (ground truth)
        t_norm = output_normalization(t)

        # Iterate through all data samples
        for d in range(len(data)):

            # normalize the inputs
            inputs_normalized = input_normalization(inputs[d, :])

            # Forward pass
            neuron_pos = get_L2_neuron_position(inputs_normalized, cmac_nb_neurons=cmac_nb_neurons, cmac_nb_inputs=cmac_nb_inputs, cmac_res=cmac_res, cmac_rf=cmac_rf, cmac_field_size=cmac_field_size)
            x = get_cmac_output(neuron_pos, cmac_weight_table, cmac_nb_neurons=cmac_nb_neurons, cmac_nb_outputs=cmac_nb_outputs)

            # Compute MSE of data sample
            MSE_sample[d, epoch] = np.square(np.subtract(t_norm[d], x)).mean()

            # Loop through L3 neurons within the window (receptive field) selected in L2
            for jk_neuron in range(cmac_nb_neurons):

                # Loop through outputs
                for i_output in range(cmac_nb_outputs):

                    # fetch index of weight in table
                    row = neuron_pos[jk_neuron][0]
                    col = neuron_pos[jk_neuron][1] 
                    wijk = cmac_weight_table[row, col, i_output] # Weight to be updated
                    increment = alpha * (t_norm[d, i_output] - x[i_output]) / cmac_nb_neurons # Increment to be added
                    new_cmac_weight_table[row, col, i_output] = wijk + increment # New weight

        if plot_table:
            if epoch == 0:
                img = plt.imshow(new_cmac_weight_table[:,:,1], interpolation="nearest")
                #plt.show()
            else:
                img.set_data(new_cmac_weight_table[:,:,1])
                plt.title('Weight table at epoch {}'.format(epoch+1))
                plt.pause(0.1)
                plt.draw()

        # Update weights for this epoch
        # new_cmac_weight_table = cmac_weight_table
        cmac_weight_table = new_cmac_weight_table

        # Print MSE of this epoch
        MSE[epoch] = MSE_sample[:, epoch].mean()
        print("MSE: " + str(MSE[epoch]))

    return new_cmac_weight_table, MSE


def save_weight_matrix():
    #####
    # this method saves in a file all data points captured so far
    #####
    print("*********")
    print("call to method: save_weight_matrix()")

    np.save(path_to_weight_matrix, np.asarray(cmac_weight_table))



if __name__=='__main__':

    # define model architecture
    input_dim = 28 * 28 # MNIST data set is 28 x 28 pixels each sample
    output_dim = 10 # MNIST are digits from 0 to 9
    num_layers = 3 # number of hidden layers
    hidden_layer_dim = 128 # number of neurons per hidden layer
    batch_size = 8 # number of samples going through the model at each iteration during batch training

    # declare model
    model = FFNN(num_inputs=input_dim, num_outputs=output_dim, num_layers=num_layers, hidden_layer_dim=hidden_layer_dim)

    # declara loss function
    loss_function = CrossEntropy()

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


    # # define cmac parameters

    # path_to_weight_matrix = '/home/bio/bioinspired_ws/src/tutorial_4/data/ffnn_weight_matrix.npy'

    # # define training dataset
    # path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_today.npy'
    # training_dataset = np.load(path_to_dataset)
    # nb_training_datapoints = 150 # set to 75 or 150 depending on the task, see tutorial description

    # # train the network
    # num_data_samples = 150
    # batch_size = 8
    # num_epochs = 50
    # plot_table = True
    # cmac_weight_table, MSE = train_cmac(cmac_weight_table, data=training_dataset[0:num_data_samples-1,:], num_epochs=num_epochs, plot_table=plot_table)

    # # save trained model
    # save_weight_matrix()

    # # Plot MSE for all epochs
    # plt.plot(range(1, num_epochs+1), MSE)
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.ylim(0, 1.1 * max(MSE))
    # plt.title("Mean squared error of training at each epoch")
    # plt.show()




