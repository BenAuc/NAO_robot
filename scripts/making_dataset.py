#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

dataset_1 = np.load('/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_v3_06-06-2022.npy') # file with 122 samples
dataset_2 = np.load('/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_v1.npy') # file with 871 samples
print(dataset_1.shape)
# shuffle dataset
idx = np.random.permutation(range(dataset_2.shape[0]))
dataset_2 = dataset_2[idx]

data = dataset_2[0:99,:] #taking only the missing 78 samples

# Combite thw datsets to one with 200 samples
sample_set = np.concatenate((dataset_1,data))
cam_y_max = 240 - 1
cam_x_max = 320 - 1

pitch_max = 2.0857
pitch_min = -2.0857
roll_max = 1.3265
roll_min = -0.3142

# saving new training set
np.save('/home/bio/bioinspired_ws/src/tutorial_4/data/data_NAO', sample_set)

print(sample_set.shape)


plt.figure()
plt.scatter(sample_set[:,0],sample_set[:,1])
plt.plot([0, cam_x_max], [cam_y_max, cam_y_max], color='red')
plt.plot([cam_x_max, cam_x_max], [0, cam_y_max], color='red')
plt.xlabel('x_cam_coordinates')
plt.ylabel('y_cam_coordinates')
plt.xlim(0,330)
plt.ylim(0,250)
plt.title('Dataset consist of 101 + 99 samples')
plt.figure()
plt.scatter(sample_set[:,2],sample_set[:,3])
plt.plot([pitch_max, pitch_max], [roll_min, roll_max], color='red')
plt.plot([pitch_min, pitch_min], [roll_min, roll_max], color='red')

plt.plot([pitch_min, pitch_max], [roll_max, roll_max], color='red')
plt.plot([pitch_min, pitch_max], [roll_min, roll_min], color='red')
plt.xlabel('pitch')
plt.ylabel('roll')
plt.title('Dataset consist of 101 + 99 samples')
plt.show()