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

def input_normalization(coordinates):
    #####
    # this method takes as input a python list 1x2 corresponding to (x,y) coordinates and normalizes it
    # Inputs:
    #  coordinates: array of dim 2 x 1 containing the (x, y) pixel coordinates
    # Outputs:
    #  array of dim 2 x 1 containing the (x, y) pixel coordinates normalized to the iamge resolution
    #####

    return [float(coordinates[0]) / cam_x_max, float(coordinates[1]) / cam_y_max]


# def get_L2_neuron_position(input_data):
#     #####
#     # this method returns the position of neurons in L2 activated by a given input
#     # Inputs:
#     #  input_data: array of dim 2 x 1 containing the (x, y) normalized pixel coordinates
#     # Outputs:
#     #  neuron_pos: array of dimensions set by the size of the receptive field and number of ouputs
#     #       containing the indices of the activated L2 neurons in the weight table
#     #####

#     # initialize variables
#     position = [] # position of single L2 neuron
#     neuron_pos = [] # positions of all L2 neurons
#     displacement_list = [] # list of displacements along input dimensions
#     quantized_ip_list = [] # list of quantized inputs
    
#     # Perform quantization step (L1)
#     for i_channel in range(cmac_nb_inputs):

#         # quantize the input per the chosen resolution
#         quantized_ip = int(input_data)[i_channel] * cmac_res

#         # safety check to force the quantization to remain within boundaries
#         if quantized_ip >= cmac_res:
#             quantized_ip = cmac_res

#         # append to the list
#         quantized_ip_list.append(quantized_ip)

#     # find coordinates of all activated L2 neurons
#     for i_neuron in range(cmac_nb_neurons):
#         position = []
        
#         # for all dimensions
#         for inputs in range(cmac_nb_inputs):

#             # compute the shift
#             shift_amount  = (cmac_field_size - quantized_ip_list[inputs]) % cmac_field_size

#             # compute local coordinates in receptive field
#             local_coord = (shift_amount  + cmac_rf[i_neuron][inputs]) % cmac_field_size

#             # compute L2 neuron coordinates in the weight tables
#             coord = quantized_ip_list[inputs] + local_coord
            
#             # append to list
#             position.append(coord) # why do we use a flat array for a set of (x,y) coordinates ? 
#             # this can work but can also be misleading

#         # append to list
#         neuron_pos.append(position)

#     print("**************")
#     print("set of L2 neurons activated :", neuron_pos)

#     return neuron_pos


def get_L2_neuron_position(input_data):
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

            # print("******************************")
            # print("neuron # :", i_neuron)
            # print("******************************")
            neuron_coord = []

            for i_channel in range(cmac_nb_inputs):

                # print("*******")
                # print("channel # :", i_channel)
                    
                input_index_q = int(input_data[i_channel] * cmac_res)
                # print("shift idx :", input_index_q)

                shift_amount_d = cmac_field_size - input_index_q % cmac_field_size
                # print("shift amount :", shift_amount_d)

                #print("neuron coordinates in rf:", self.cmac_rf[i_neuron][i_channel])
                local_coord_p = (shift_amount_d + cmac_rf[i_neuron][i_channel]) % cmac_field_size
                #print("local coordinates :", local_coord_p)

                coord = input_index_q + local_coord_p

                if coord >= cmac_res:
                    coord = cmac_res-1

                neuron_coord.append(coord)

            position.append(neuron_coord)

        #print("**************")
        #print("set of L2 neurons activated :", position)
        
        return position


def get_cmac_output(neuron_pos):
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

            # print("row :", row)
            # print("col :", col)

            # if row >= cmac_res:
            #     row = cmac_res-1
            #     print("correction : ", row)

            # if col >= cmac_res:
            #     col = cmac_res-1
            #     print("correction : ", col)

            # Add weight from weight table
            x[i_output] = x[i_output] + cmac_weight_table[row, col, i_output]

    # # check whether joints are within their limit range and if not enforce it
    # # check left shoulder pitch
    # if x[0] > (1 - joint_limit_safety_f) * l_shoulder_pitch_limits[1]:
    #     print("Joint state pitch mapped out of its limits. Correction applied.", (1 - joint_limit_safety_f) * l_shoulder_pitch_limits[1])
    #     x[0] = (1 - joint_limit_safety_f) * l_shoulder_pitch_limits[1]

    # elif x[0] < (1 - joint_limit_safety_f) * l_shoulder_pitch_limits[0]:
    #     print("Joint state pitch mapped out of its limits. Correction applied.", (1 - joint_limit_safety_f) * l_shoulder_pitch_limits[0])
    #     x[0] = (1 - joint_limit_safety_f) * l_shoulder_pitch_limits[0]

    # # check left shoulder roll
    # if x[1] > (1 - joint_limit_safety_f) * l_shoulder_roll_limits[1]:
    #     print("Joint state roll mapped out of its limits. Correction applied.", (1 - joint_limit_safety_f) * l_shoulder_roll_limits[1])
    #     x[1] = (1 - joint_limit_safety_f) * l_shoulder_roll_limits[1]

    # elif x[1] < (1 - joint_limit_safety_f) * l_shoulder_roll_limits[0]:
    #     print("Joint state roll mapped out of its limits. Correction applied.", (1 - joint_limit_safety_f) * l_shoulder_roll_limits[0])
    #     x[1] = (1 - joint_limit_safety_f) * l_shoulder_roll_limits[0]

    return x

def train_cmac(cmac_weight_table, data, num_epochs):
    # Train the CMAC
    # Inputs:
    #   cmac_weight_table: untrained weight table
    #   data: training data of dim N samples x 4 where each sample is a (x, y) pixel coordinates 
    #       followed by (shoulder_pitch, shoulder_roll) joint state
    #   num_epochs: number of epochs for the training
    # Outputs:
    #   new_cmac_weight_table: trained weight table
    # Example call:
    #   cmac_weight_table = train_cmac(cmac_weight_table, training_dataset[0:149,:], 10)

    # Initialize variables
    new_cmac_weight_table = np.zeros(cmac_weight_table.shape) # Trained weight table
    alpha = 0.02 # Learning rate
    inputs = data[:, 0:] # Inputs
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

        inputs = data[:, 0:] # Inputs
        t = data[:,-2:] # Targets (ground truth)

        # Iterate through all data samples
        for d in range(len(data)):

            # normalize the inputs
            inputs_normalized = input_normalization(inputs[d, :])

            # Forward pass
            neuron_pos = get_L2_neuron_position(inputs_normalized)
            x = get_cmac_output(neuron_pos)

            # Compute MSE of data sample
            MSE_sample[d, epoch] = np.square(np.subtract(t[d],x)).mean()

            # Loop through L3 neurons within the window (receptive field) selected in L2
            for jk_neuron in range(cmac_nb_neurons):

                # Loop through outputs
                for i_output in range(cmac_nb_outputs):

                    # fetch index of weight in table
                    row = neuron_pos[jk_neuron][0]
                    col = neuron_pos[jk_neuron][1] 
                    wijk = cmac_weight_table[row, col, i_output] # Weight to be updated
                    increment = alpha * (t[d, i_output] - x[i_output]) / cmac_nb_neurons # Increment to be added
                    new_cmac_weight_table[row, col, i_output] = wijk + increment # New weight

        # Update weights for this epoch
        new_cmac_weight_table = cmac_weight_table

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

# define camera resolution
# pixels idx run from 0 to resolution - 1
cam_y_max = 240 - 1
cam_x_max = 320 - 1

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
num_epochs = 150
cmac_weight_table, MSE = train_cmac(cmac_weight_table, training_dataset[0:num_data_samples-1,:], num_epochs)

print("weight matrix : ", cmac_weight_table)

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

print("bins : ", bins)

plt.hist(np.ndarray.flatten(cmac_weight_table))
plt.xlabel("Binned weight value total " + str(len(bins) - 1) + " bins")
plt.ylabel("# of counts")
plt.title("Histogram of the weight distribution")
plt.show()





 
