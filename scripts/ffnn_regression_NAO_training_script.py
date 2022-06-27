#!/usr/bin/env python

#################################################################
# file name: ffnn_regression_NAO_training_script.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 23-06-2022
# last edit: 23-06-2022 (Benoit, Vildana)
# function: trains a feed-forward neural network on the acquired data to control
# NAO's shoulder joint and saves the final model
# inputs: self-acquired training data set
# outputs: pickle file that contains the weight matrices, training statistics and the loss/accuracy curves
#################################################################

from ffnn_model import FFNN, MSE, Linear
import numpy as np
from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':

    # define model architecture
    input_dim = 2  # (x,y) pixel coordinates
    output_dim = 2  # 2 degrees of freedom
    num_layers = 2  # number of all layers (excluding the inputs)
    hidden_layer_dim = 8 # number of neurons per hidden layer
    epochs = 20 # number of epochs
    batch_size = 1  # number of samples going through the model at each iteration during batch training
    step_size = 0.1 # Gradient descent step size alpha
    loss_function = MSE()  # instantiate loss function

    # parameters to save model
    save_model = True
    save_weight_path = "/home/bio/bioinspired_ws/src/tutorial_4/data/model_weights_NAO/weight_matrix.pickle"  # path to the mnist data folder
    save_statistics_path = "/home/bio/bioinspired_ws/src/tutorial_4/data/model_weights_NAO/fnn_statistics.pickle" 
    save_normalization_path = "/home/bio/bioinspired_ws/src/tutorial_4/data/model_weights_NAO/fnn_normalization.pickle" 
    save_normalized_training_set_path = "/home/bio/bioinspired_ws/src/tutorial_4/data/model_weights_NAO/fnn_training_set.pickle" 
    
    ### get train and test sets ###
    dataset = np.load('/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_set/training_data_NAO_200_points_FINAL.npy')
    np.random.shuffle(dataset)
    print(dataset.shape)

    # spliting into input (X) and output (Y)
    trainX_raw = dataset[0:150,0:2]
    trainY_raw = dataset[0:150,2:4]
    testX_raw = dataset[150:,0:2]
    testY_raw = dataset[150:,2:4]

    print("trainX_raw shape :", trainX_raw.shape)
    print("trainY_raw shape:", trainY_raw.shape)
    print("testX_raw shape:", testX_raw.shape)
    print("testY_raw shape:", testY_raw.shape)
    
    # define min and max for normalization of input space
    cam_y_max = 240 - 1
    cam_x_max = 320 - 1

    # define min and max for normalization of output space
    max_pitch = np.amax(trainY_raw[:,0]) # 0.25  # 0.06
    min_pitch = np.amin(trainY_raw[:,0])# -1 # -0.63
    max_roll = np.amax(trainY_raw[:,1])# 0.5 # 0.39
    min_roll = np.amin(trainY_raw[:,1])# -0.4 # -0.32

    pitch_limits = np.array([min_pitch,max_pitch])
    roll_limits = np.array([min_roll,max_roll])
    pitch_range = pitch_limits[1] - pitch_limits[0]
    roll_range = roll_limits[1] - roll_limits[0]

    # normalization of dataset
    trainX = trainX_raw/np.array([cam_x_max,cam_y_max])
    trainY = (trainY_raw - np.array([pitch_limits[0], roll_limits[0]]))/ np.array([pitch_range, roll_range])
    testX = testX_raw/np.array([cam_x_max,cam_y_max])
    testY = (testY_raw - np.array([pitch_limits[0], roll_limits[0]]))/ np.array([pitch_range, roll_range])
    
    # Batchs
    num_batches = int(np.floor(trainY.shape[0] / batch_size))
    BX = [trainX[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainX
    BY = [trainY[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainY

    # instantiate model
    model = FFNN(num_inputs=input_dim, num_outputs=output_dim, num_layers=num_layers, hidden_layer_dim=hidden_layer_dim, activation_func=Linear(), load_model=False, model_params={})

    # Initialize stats arrays
    loss_test = np.zeros([epochs, 1])
    acc_test = np.zeros([epochs, 1])
    loss_train = np.zeros([epochs, 1])
    acc_train = np.zeros([epochs, 1])

    #Loop through data
    batch_index = range(len(BX))

    for epoch in range(epochs):
        print("epoch # :", epoch)

        # train loss and accuracy of whole train set
        output = model.forward(trainX)
        y_pred = model.predict(trainX)
        loss_train[epoch] = loss_function.forward(output, trainY)
        acc_train[epoch] = np.mean(y_pred == trainY)

        print("epoch train loss :", loss_train[epoch])

        # test loss and accuracy of whole test set
        output_test = model.forward(testX)
        y_pred_test = model.predict(testX)
        loss_test[epoch] = loss_function.forward(output_test, testY)
        acc_test[epoch] = np.mean(y_pred_test == testY)

        print("epoch test loss :", loss_test[epoch])
        np.random.shuffle(batch_index)

        for idx in range(int(len(BX))):
            
            # pick batch among shuffled indices
            batch = batch_index[idx]

            # Take the set from current batch
            X = BX[batch]
            Y = BY[batch]
            model.zero_grad()

            #Forward
            output = model.forward(X)

            # batch training loss and the derivative
            loss = loss_function.forward(model_output = output, ground_truth = Y)

            d_loss = loss_function.backward(output, Y)
            

            # if batch % 5 == 0:
            #     print("batch # " + str(batch))
            #     print("loss is : ", loss)
            #     print("loss gradient is : ", d_loss)
            #     print("model output is : ", output)
            #     print("ground truth is : ", Y)

            # backward propagation
            grads = model.backward(d_loss)

            # gradient descent step
            model.step(step_size=step_size)

        # shuffle data sets and regenerate list of batches
        # np.random.shuffle(trainX)
        # np.random.shuffle(trainY)
        # BX = [trainX[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainX
        # BY = [trainY[(i * batch_size):((i + 1) * batch_size)] for i in range(num_batches)]  # list of batches trainY

    loss_final = {'loss_train':loss_train, 'loss_test':loss_test}

    normalization_output = {'max_pitch':max_pitch, 'min_pitch':min_pitch, 'max_roll':max_roll, 'min_roll': min_roll}
    
    training_set = {'input_x_y': trainX, 'output_pitch_roll': trainY}

    # save mode
    if save_model == True:

        with open(save_normalized_training_set_path, 'wb') as handle:
            pickle.dump(training_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(save_normalization_path, 'wb') as handle:
            pickle.dump(normalization_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(save_weight_path, 'wb') as handle:
            pickle.dump(model.model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(save_statistics_path, 'wb') as handle:
            pickle.dump(loss_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training completed")
    print("************")
    # print("training accuracy :", acc_train)
    # print("training loss :", loss_train)

    # plot training loss
    font_size = 18
    fig, ax = plt.subplots(1,1)
    ax.set_title("Loss over epochs for training and test datasets", fontsize=font_size)
    
    ax.plot(list(range(1,epochs+1)),loss_train, color = 'r', label='loss_train', linewidth=2)
    ax.plot(list(range(1,epochs+1)),loss_test, color = 'b', label='loss_test')
    
    ax.set_ylabel("MSE loss", fontsize=font_size)
    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_xticks(range(1,epochs+1))
    
    ax.legend()
    plt.show()

    # plot output of model on training set and ground truth
    YY = model.forward(trainX)
    fig, ax = plt.subplots(1,1)
    ax.set_title("Comparison of model output and ground truth", fontsize=font_size)
    
    ax.scatter(YY[:,0], YY[:,1], c='g', label='model output')
    ax.scatter(trainY[:,0], trainY[:,1], c='r', label='ground truth')

    ax.set_ylabel("normalized roll angle", fontsize=font_size)
    ax.set_xlabel("normalized pitch angle", fontsize=font_size)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.legend()
    plt.show()

    # plot distance between pair of data points
    # each pair of points being the output of model on training set and ground truth

    output = model.forward(trainX)
    truth = trainY

    #output = model.forward(testX)
    #truth = testY

    n_samples = output.shape[0]

    fig, ax = plt.subplots(1,1)
    ax.set_title("Plot of distance between model output and ground truth data points (test set)", fontsize=font_size)

    for i_sample in range(n_samples):
        ax.plot([output[i_sample,0], truth[i_sample,0]], [output[i_sample,1], truth[i_sample,1]], marker='o')

    ax.set_ylabel("normalized roll angle", fontsize=font_size)
    ax.set_xlabel("normalized pitch angle", fontsize=font_size)
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.legend()
    plt.show()

    print(training_set)


