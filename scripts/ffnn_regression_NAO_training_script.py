
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

from ffnn_model import FFNN, MSE
import numpy as np
from matplotlib import pyplot as plt
import pickle

if __name__ == '__main__':

    # define model architecture
    input_dim = 2  # (x,y) pixel coordinates
    output_dim = 2  # 2 degrees of freedom
    num_layers = 3  # number of all layers (excluding the inputs)
    hidden_layer_dim = 16  # number of neurons per hidden layer
    epochs = 5  # number of epochs
    batch_size = 30  # number of samples going through the model at each iteration during batch training
    step_size = 0.05 # Gradient descent step size alpha
    loss_function = MSE()  # instantiate loss function
    mnist_path = "../data/mnist-data"  # path to the mnist data folder
    save_weight_path = "../data/model_weights_NAO/weight_matrix_final.pickle"  # path to the mnist data folder
    save_statistics_path = "../data/model_weights_NAO/fnn_statistics_final.pickle" 
    regularisation = 0.1
    pitch_limits = np.array([-2.0857, 2.0857])
    roll_limits = np.array([-0.3142, 1.3265])
    pitch_range = pitch_limits[1] - pitch_limits[0]
    roll_range = roll_limits[1] - roll_limits[0]


    # get train and test sets
    dataset = np.load("../data/NAO_training_data.npy")
    trainX_raw = dataset[:,0:2]
    trainY_raw = dataset[:,2:4]

    # normalize input and output
    cam_y_max = 240 - 1
    cam_x_max = 320 - 1

    trainX = trainX_raw/np.array([239,319])
    trainY = (trainY_raw - np.array([pitch_limits[0], roll_limits[0]]))/ np.array([pitch_range, roll_range])


    # Batchs
    num_batches = int(np.floor(trainY.shape[0] / batch_size))
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

        # train and test loss of whole test set

        #output_test = model.forward(testX)
        #loss_test[epoch] = loss_function.forward(output_test, testY)

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
    fig, ax = plt.subplots(1,1)

    ax.set_title("Loss over epochs for training and test datasets", fontsize=font_size)
    ax.plot(list(range(1,epochs+1)),loss_train, color = 'r', label='loss_train', linewidth=2)
    #ax[0].plot(list(range(epochs)),loss_test, color = 'b', label='loss_test')
    ax.set_ylabel("MSE loss", fontsize=font_size)
    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_xticks(range(1,epochs+1))
    ax.legend()

    fig.show()