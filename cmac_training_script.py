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
import numpy as np
import copy
import matplotlib.pyplot as plt

'''
this method takes as input a python list 1x2 corresponding to (x,y) coordinates and normalizes it
Inputs:coordinates: array of dim 2 x 1 containing the (x, y) pixel coordinates
Outputs:array of dim 2 x 1 containing the (x, y) pixel coordinates normalized to the image resolution
'''
cam_y_max = 240 - 1
cam_x_max = 320 - 1

def input_normalization(input):
    normalised_data = (input - np.min(input)) / (np.max(input) - np.min(input))
    return normalised_data


'''
this method returns the position of neurons in L2 activated by a given input
 Inputs:input_data: array of dim 2 x 1 containing the (x, y) normalized pixel coordinates
Outputs:neuron_pos: array of dimensions set by the size of the receptive field n_a and number of ouputs n_x
contains the indices of the activated L2 neurons in the weight table
'''


def get_L2_neuron_position(input_data):
    position = []

    for i_neuron in range(cmac_nb_neurons):

        neuron_coord = []

        for i_channel in range(cmac_nb_inputs):
            input_index_q = int(input_data[i_channel] * cmac_res)
            shift_amount_d = cmac_field_size - input_index_q % cmac_field_size
            local_coord_p = (shift_amount_d + cmac_rf[i_neuron][i_channel]) % cmac_field_size
            coord = input_index_q + local_coord_p
            if coord >= cmac_res:
                coord = cmac_res - 1
            neuron_coord.append(coord)
        position.append(neuron_coord)

    return position


'''
Calculate the ouput of the CMAC after L3
Inputs:neuron_pos: list of indices of the neurons within the receptive field, computed in L2. Can be viewed as activtion address vector
Outputs:x: list of values, each corresponding to an output of the CMAC
'''


def get_cmac_output(neuron_pos, cmac_weight_table):
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


'''
Train the CMAC
Inputs:
cmac_weight_table: untrained weight table
data: training data of dim N samples x 4 where each sample is a (x, y) pixel coordinates followed by (shoulder_pitch, shoulder_roll) joint state
num_epochs: number of epochs for the training
Outputs:
new_cmac_weight_table: trained weight table

'''


def train_cmac(cmac_weight_table, input, output, num_epochs):
    # Iterate through all data samples
    for d in range(len(input)):

        # Forward pass
        neuron_pos = get_L2_neuron_position(input[d, :])
        x = get_cmac_output(neuron_pos, cmac_weight_table)

        # Compute MSE of data sample
        MSE_sample[d, num_epochs] = np.square(np.subtract(output[d], x)).mean()

        for i_output in range(cmac_nb_outputs):
            increment = alpha * (output[d, i_output] - x[i_output]) / cmac_nb_neurons  # Increment to be added
            for jk_neuron in range(cmac_nb_neurons):
                row = neuron_pos[jk_neuron][0]
                col = neuron_pos[jk_neuron][1]
                wijk = cmac_weight_table[row, col, i_output]  # Weight to be updated
                cmac_weight_table[row, col, i_output] = wijk + increment  # New weight
        # Print MSE of this epoch
        MSE = MSE_sample[:, num_epochs].mean()

    return cmac_weight_table, MSE


'''
Testing the model offline
'''


def test_cmac(cmac_weight_table, input, output):
    mse_test = []

    for d in range(len(input)):
        # normalize the inputs
        # inputs_normalized = input_normalization(input[d, :])

        # Forward pass
        neuron_pos = get_L2_neuron_position(input[d, :])
        x = get_cmac_output(neuron_pos, cmac_weight_table)

        # Compute MSE of data sample
        mse_test.append(np.square(np.subtract(output[d], x)).mean())

    return np.mean(mse_test)


'''
Main code
'''

# define cmac parameters
cmac_nb_inputs = 2
cmac_nb_outputs = 2
cmac_res = 50  # resolution
cmac_field_size = 5  # receptive field size: set to 3 or 5 depending on the task, see tutorial description
cmac_nb_neurons = cmac_field_size  # to be defined, max field_size x field_size
cmac_rf = [[0, 3], [1, 0], [2, 2], [3, 4], [4, 1]]
cmac_weight_table = np.random.random_sample(
    (cmac_res, cmac_res, cmac_nb_outputs))  # Not all entries correspond to a neuron, depends on cmac_nb_neurons
cmac_weight = np.zeros([cmac_res, cmac_res, cmac_nb_outputs])
# path_to_weight_matrix = '/home/bio/bioinspired_ws/src/tutorial_4/data/cmac_weight_matrix.npy'

# define camera resolution
# pixels idx run from 0 to resolution - 1
cam_y_max = 240 - 1
cam_x_max = 320 - 1

# define joint limits
l_shoulder_pitch_limits = [-2.0857, 2.0857]
l_shoulder_roll_limits = [-0.3142, 1.3265]
joint_limit_safety_f = 0.05

# define training dataset
path_to_dataset = 'C:\\Users\\vilda\\Documents\\GitHub\\bilhr_dream_team\\data\\training_data_today.npy'
training_dataset = np.load(path_to_dataset)
input_data = training_dataset[:, 0:2]
output_data = training_dataset[:, -2:]

# Normalize the data
input_data = input_normalization(input_data)
output_data = input_normalization(output_data)
num_epochs = 10
alpha = 0.2  # Learning rate
MSE_sample = np.zeros((training_dataset.shape[0], num_epochs))  # MSE of each sample at every epoch
MSE_train_list = [0] * num_epochs  # General MSE at every epoch
MSE_err_list = [0] * num_epochs
num_samples = 102  # for training, validation would be 120-this value


# Shuffle the training set in each epoch
per = np.random.permutation(len(input_data))
input_data = input_data[per]
output_data = output_data[per]

# train, validation split
X_train = input_data[0:num_samples]
Y_train = output_data[0:num_samples]
X_test = input_data[num_samples:]
Y_test = output_data[num_samples:]
# iterate over epochs
for i in range(num_epochs):
    cmac_weight_table, MSE_train = train_cmac(cmac_weight_table, X_train, Y_train, i)
    MSE_train_list[i] = MSE_train
    MSE_err = test_cmac(cmac_weight_table, X_test, Y_test)
    MSE_err_list[i] = MSE_err

# Plot MSE for all epochs
fig, axes = plt.subplots(1, 1)
axes.plot(MSE_train_list)
axes.plot(MSE_err_list)
axes.set_xlabel("Epoch")
axes.set_ylabel("MSE")
axes.set_ylim(0, 1.1 * max(MSE_err_list))

plt.show()

# testing

mse_test = test_cmac(cmac_weight_table, X_test, Y_test)
print('Test mse of the model is {}'.format(mse_test))
