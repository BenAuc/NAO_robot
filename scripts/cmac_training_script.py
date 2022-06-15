#!/usr/bin/env python

#################################################################
# file name: cmac_training_script.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 06-06-2022
# last edit: 07-06-2022 (Benoit)
# function: trains the cmac and saves the weight matrix
# inputs: design of the cmac
# outputs: .npy file that contains the weight matrix
#################################################################

from random import random
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
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


def get_L2_neuron_position_old(input_data, cmac_nb_neurons, cmac_nb_inputs, cmac_res, cmac_rf, cmac_field_size):
    #     #####
    #     # this method returns the position of neurons in L2 activated by a given input
    #     # Inputs:
    #     #  input_data: array of dim 2 x 1 containing the (x, y) normalized pixel coordinates
    #     # Outputs:
    #     #  neuron_pos: array of dimensions set by the size of the receptive field n_a and number of ouputs n_x
    #     #       contains the indices of the activated L2 neurons in the weight table
    #     #####

        position = []

        for i_neuron in range(cmac_nb_neurons):
            neuron_coord = []

            for i_channel in range(cmac_nb_inputs):
                    
                input_index_q = int(input_data[i_channel] * cmac_res)

                shift_amount_d = cmac_field_size - input_index_q % cmac_field_size

                local_coord_p = (shift_amount_d + cmac_rf[i_neuron][i_channel]) % cmac_field_size

                # local_coord_p = cmac_rf[i_neuron][i_channel]

                coord = input_index_q + local_coord_p

                if coord >= cmac_res:
                    coord = cmac_res-1
                if coord < 0:
                    coord = 0

                neuron_coord.append(coord)

            position.append(neuron_coord)
        
        return position

def get_L2_neuron_position(input_data, cmac_nb_neurons, cmac_nb_inputs, cmac_res, cmac_rf, cmac_field_size):
    #####
    # this method returns the position of neurons in L2 activated by a given input
    # Inputs:
    #   input_data: array of dim 2 x 1 containing the (x, y) normalized pixel coordinates
    # Outputs:
    #   neuron_pos: array of dimensions set by the size of the receptive field n_a and number of ouputs n_x
    #       contains the indices of the activated L2 neurons in the weight table
    #####


def get_cmac_output(neuron_pos, cmac_weight_table, cmac_nb_neurons, cmac_nb_outputs):
    # Calculate the ouput of the CMAC after L3
    # Inputs:
    #   neuron_pos: list of indices of the neurons within the receptive field, computed in L2. Can be viewed as activtion address vector
    # Outputs:
    #   x: list of values, each corresponding to an output of the CMAC

    # Initialize the outputs to 0
    x = [0] * cmac_nb_outputs

    # Loop through L3 neurons within the window (receptive field) selected in L2
    for jk_neuron in range(cmac_nb_neurons):

        # Loop through outputs
        for i_output in range(cmac_nb_outputs):

            # fetch index of weight in table
            row = neuron_pos[jk_neuron][0]
            col = neuron_pos[jk_neuron][1] 

            # Add weight from weight table
            x[i_output] = x[i_output] + cmac_weight_table[row, col, i_output]

    return x


def train_cmac(cmac_weight_table, data, num_epochs, plot_table):
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

    # define cmac parameters
    cmac_nb_inputs = 2
    cmac_nb_outputs = 2
    cmac_res = 50 # resolution
    cmac_field_size = 5 # receptive field size: set to 3 or 5 depending on the task, see tutorial description
    cmac_nb_neurons = cmac_field_size # to be defined, max field_size x field_size
    path_to_weight_matrix = '/home/bio/bioinspired_ws/src/tutorial_4/data/cmac_weight_matrix.npy'

    # receptive field: see additional material 3 on moodle. Here 5 neurons with coordinates in shape of a cross.
    cmac_rf = [[0, 3], [1, 0], [2, 2], [3, 4], [4, 1]] 
    #cmac_rf = [[0, 0], [1, 1], [2, 2]] 
    cmac_weight_table = np.random.normal(-0.25, 0.25, (cmac_res, cmac_res, cmac_nb_outputs)) # Not all entries correspond to a neuron, depends on cmac_nb_neurons
    # cmac_weight_table = np.random.uniform(-0.2, 0.2, (cmac_res, cmac_res, cmac_nb_outputs)) # Not all entries correspond to a neuron, depends on cmac_nb_neurons

    # define joint limits
    l_shoulder_pitch_limits = [-2.0857, 2.0857]
    l_shoulder_roll_limits = [-0.3142, 1.3265]
    joint_limit_safety_f = 0.05

    # define training dataset
    path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_today.npy'
    training_dataset = np.load(path_to_dataset)
    nb_training_datapoints = 150 # set to 75 or 150 depending on the task, see tutorial description

    # train the CMAC network
    num_data_samples = 150
    num_epochs = 50
    plot_table = True
    cmac_weight_table, MSE = train_cmac(cmac_weight_table, data=training_dataset[0:num_data_samples-1,:], num_epochs=num_epochs, plot_table=plot_table)
    plt.imshow(cmac_weight_table[:,:,1], interpolation="nearest")
    plt.show()

    # print("weight matrix : ", cmac_weight_table)

    # save cmac training
    save_weight_matrix()

    # Plot MSE for all epochs

    plt.plot(range(1, num_epochs+1), MSE)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.ylim(0, 1.1 * max(MSE))
    plt.title("Mean squared error of training at each epoch")
    plt.show()

    # Plot weight distribution
    bin_size = 0.04
    bins = np.arange(np.amin(cmac_weight_table), np.amax(cmac_weight_table) + bin_size, bin_size)
    distribution = np.histogram(cmac_weight_table, bins=bins)[0]

    # print("bins : ", bins)

    plt.hist(np.ndarray.flatten(cmac_weight_table))
    plt.xlabel("Binned weight value total " + str(len(bins) - 1) + " bins")
    plt.ylabel("# of counts")
    plt.title("Histogram of the weight distribution")
    plt.show()


    ## Make surf plot from a forward pass
    # Compute forward pass values
    print("Please wait, computing CMAC outputs...")
    cam_y_max = 240 - 1
    cam_x_max = 320 - 1
    x = np.linspace(0, cam_x_max, 100).T       # Generate artificial x coordinates
    y = np.linspace(0, cam_y_max, 100).T       # Generate artificial y coordinates
    # inputs = np.hstack((x, y))                 # Summarize artificail inputs
    X, Y = np.meshgrid(x, y)                   # Generate mesh for later surf plot
    pitch = np.zeros(X.shape)                  # Initialize pitch matrix
    roll = np.zeros(X.shape)                   # Initialize roll matrix
    for x_coordinate in range(len(x)):
        for y_coordinate in range(len(y)):
            sel_input = [x[x_coordinate], y[y_coordinate]] # Select input sample
            inputs_normalized = input_normalization(sel_input) # normalize the inputs
            # Forward pass
            neuron_pos = get_L2_neuron_position(inputs_normalized, cmac_nb_neurons=cmac_nb_neurons, cmac_nb_inputs=cmac_nb_inputs, cmac_res=cmac_res, cmac_field_size=cmac_field_size)
            output = get_cmac_output(neuron_pos, cmac_weight_table=cmac_weight_table, cmac_nb_neurons=cmac_nb_neurons, cmac_nb_outputs=cmac_nb_outputs) # Renamed "x" to "output" to prevent naming conflicts
            # Store pitch and roll
            pitch[y_coordinate, x_coordinate] = output[0]
            roll[y_coordinate, x_coordinate] = output[1]
    print("Computation completed")

    # Make surf plots
    fig = plt.figure()
    # Plot the pitch
    ax1 = fig.add_subplot(211, projection="3d")
    ax1.plot_surface(X, Y, pitch)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("Pitch")
    # Plot the roll
    ax2 = fig.add_subplot(212, projection="3d")
    ax2.plot_surface(X, Y, roll)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Roll")

    plt.show()



