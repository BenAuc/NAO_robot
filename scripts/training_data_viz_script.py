#!/usr/bin/env python

#################################################################
# file name: training_data_viz_script.py
# author's name: Benoit Auclair
# created on: 02-06-2022
# last edit: 09-06-2022 (Diego)
# purpose: visualize the data points acquired to train the cmac
# inputs: path(s) to the dataset(s) and number of data points to visualize
# outputs: 2 figures that show the distribution of the inputs and outputs
#################################################################

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_today.npy'
training_dataset = np.load(path_to_dataset)

# second data set if we want to overlay and compare
# path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_v3.npy'
# training_dataset2 = np.load(path_to_dataset)

nb_datapoints = 150 # set to 75 or 150 depending on the task, see tutorial description

#####
# plot camera pixel distribution

x = training_dataset[:nb_datapoints, 0]
y = training_dataset[:nb_datapoints, 1]

cam_y_max = 240 - 1
cam_x_max = 320 - 1

# second data set if we want to overlay and compare
# plt.plot(training_dataset2[:, 0], training_dataset2[:, 1], 'o', color='black')
plt.plot(x, y, 'x', color='green')

# plot limits of input space
plt.plot([0, cam_x_max], [cam_y_max, cam_y_max], color='red')
plt.plot([cam_x_max, cam_x_max], [0, cam_y_max], color='red')

plt.xlabel("x pixel coordinate")
plt.ylabel("y pixel coordinate")
plt.title("Training data distribution for first " + str(nb_datapoints) + " points")
plt.show()

#####
# plot joint state distribution

pitch = training_dataset[:nb_datapoints, 2]
roll = training_dataset[:nb_datapoints, 3]

pitch_max = 2.0857
pitch_min = -2.0857
roll_max = 1.3265
roll_min = -0.3142

plt.plot(pitch, roll, 'x', color='green')

# plot limits of output space
plt.plot([pitch_max, pitch_max], [roll_min, roll_max], color='red')
plt.plot([pitch_min, pitch_min], [roll_min, roll_max], color='red')

plt.plot([pitch_min, pitch_max], [roll_max, roll_max], color='red')
plt.plot([pitch_min, pitch_max], [roll_min, roll_min], color='red')

plt.xlabel("pitch angle (rad)")
plt.ylabel("roll angle (rad)")
plt.title("Joint state distribution for all " + str(training_dataset.shape[0]) + " points")
plt.show()


# Surface plot of the aquired data
fig = plt.figure()
# Plot the pitch
ax1 = fig.add_subplot(211, projection="3d")
ax1.plot_trisurf(x, y, pitch)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("Pitch")
# Plot the roll
ax2 = fig.add_subplot(212, projection="3d")
ax2.plot_trisurf(x, y, roll)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("Roll")

plt.show()
