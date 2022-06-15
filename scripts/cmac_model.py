import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CMAC_Model:

    def __init__(self, num_inputs, num_outputs, res, receptive_field_size):
        # Initialize model parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.res = res
        self.receptive_field_size = receptive_field_size

        self.neural_field = self.L2_neurons_mask(self.receptive_field_size, self.res)

    def L2_neurons_mask(self, receptive_field_size, res):
        # Create a boolean mask of the L2 neuron field according to randomly initialized receptive field

        # Initialize receptive field randomly
        np.random.seed(33)
        rec_field_row = np.random.permutation(receptive_field_size)
        np.random.seed(40)
        rec_field_col = np.random.permutation(receptive_field_size)

        receptive_field = np.zeros((receptive_field_size, receptive_field_size))
        receptive_field[rec_field_row, rec_field_col] = 1
        field_size = np.around(float(res)/float(receptive_field_size))
        neural_field = np.tile(receptive_field, (field_size, field_size))
        neural_field = neural_field[0:res, 0:res]

        return neural_field

    def L1_mask(self, input_data, res, receptive_field_size, num_inputs):
        # Quantize the input data

        L1 = np.zeros((res, num_inputs))
        half_rec = receptive_field_size/2
        quantized_input = [x * res - 1 for x in input_data]
        for i in range(num_inputs):
            L1[int(quantized_input[i] - half_rec):int(quantized_input[i] + half_rec + 1), i] = 1
        
        return L1

    def active_L2_neurons(self, res, L1, neural_field):
        # Compute active L2 neurons within receptive field

        L2 = np.zeros((res,res))
        for row in range(res):
            for col in range(res):
                if neural_field[row, col] == 1:
                    if L1.T[1, col] == L1[row, 0] == 1:
                        L2[row, col] = 1

        return L2

    def compute_L3(self, L2, num_outputs, weight_matrix):
        # Compute the outputs of the CMAC (called 'x' in the thesis)

        L3 = np.zeros(num_outputs)
        for i in range(num_outputs):
            L3[i] = np.sum(np.multiply(weight_matrix[:, :, i], L2))

        return L3

    def predict(self, input_data, weight_matrix):
        # Make a forward pass on the entire CMAC

        L1 = self.L1_mask(input_data, self.res, self.receptive_field_size, self.num_inputs)
        L2 = self.active_L2_neurons(self.res, L1, self.neural_field)
        L3 = self.compute_L3(L2, self.num_outputs, weight_matrix)

        return L2, L3

    def train(self, input_data, output_data, num_epochs, alpha, plot_table):
        # Train the CMAC
        # Inputs:
        #   weight_matrix: untrained weight table
        #   data: training data of dim N samples x 4 where each sample is a (x, y) pixel coordinates 
        #       followed by (shoulder_pitch, shoulder_roll) joint state
        #   num_epochs: number of epochs for the training
        #   alpha: learning rate
        #   res: cmac resolution
        #   plot_table: flag to plot the weight table after each epoch
        # Outputs:
        #   weight_matrix: trained weight table

        # Initialize variables
        weight_matrix = np.random.normal(-0.025, 0.025, (self.res, self.res, self.num_outputs))  # Initialize weight table
        MSE_sample = np.zeros((input_data.shape[0], num_epochs)) # MSE of each sample at every epoch
        MSE = [0] * num_epochs # General MSE at every epoch

        # Delete values that do not correspond to a neuron
        for i in range(self.num_outputs):
            weight_matrix[:, :, i] = np.multiply(weight_matrix[:, :, i], self.neural_field)

        # Target training:
        print('******')
        print("Starting target training...")

        # Permutate training samples
        per = np.random.permutation(len(input_data))
        input_data = input_data[per]
        output_data = output_data[per]
        
        # Repeat num_epochs times
        for epoch in range(num_epochs):

            # Iterate through all data samples
            for d in range(len(input_data)):

                # Forward pass
                L2, output_pred = self.predict(input_data=input_data[d], weight_matrix=weight_matrix)

                # Compute MSE of data sample
                MSE_sample[d, epoch] = np.square(np.subtract(output_data[d], output_pred)).mean()

                # Loop through outputs
                for i in range(self.num_outputs):
                    weight_matrix[:, :, i] += alpha * L2 / self.receptive_field_size * (output_data[d, i] - output_pred[i])

            if plot_table:
                if epoch == 0:
                    img = plt.imshow(weight_matrix[:,:,1], interpolation="nearest")
                    #plt.show()
                else:
                    img.set_data(weight_matrix[:,:,1])
                    plt.title('Weight table at epoch {}'.format(epoch+1))
                    plt.pause(0.1)
                    plt.draw()


            # Print MSE of this epoch
            print("Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
            MSE[epoch] = MSE_sample[:, epoch].mean()
            print("MSE: " + str(MSE[epoch]))

        return weight_matrix, MSE

def input_normalization(coordinates):
    cam_y_max = 240 - 1
    cam_x_max = 320 - 1
    return np.array(coordinates, dtype=float) / np.array([cam_x_max, cam_y_max], dtype=float)

def output_normalization(output_data):
    max_pitch = 0.06
    min_pitch = -0.63
    max_roll = 0.39
    min_roll = -0.32

    return (output_data - np.array([min_pitch, min_roll], dtype=float)) / (np.array([max_pitch, max_roll]) - np.array([min_pitch, min_roll], dtype=float))

def output_denormalization(output_data):
    max_pitch = 0.06
    min_pitch = -0.63
    max_roll = 0.39
    min_roll = -0.32

    return output_data  * (np.array([max_pitch, max_roll], dtype=float) - np.array([min_pitch, min_roll], dtype=float)) + np.array([min_pitch, min_roll], dtype=float)

if __name__ == '__main__':
    receptive_field_size = 5
    cmac = CMAC_Model(num_inputs=2, num_outputs=2, res=50, receptive_field_size=receptive_field_size)

    data = np.load('/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_v1.npy')
    num_samples = 150 # Take subset of all data (in total more than 150)
    data = np.random.shuffle(data) # Shuffle the data to avoid taking always the same datasets
    data = data[0:num_samples]
    input_data = data[:, 0:2] # Inputs
    output_data = data[:,-2:] # Targets (ground truth)

    # Normalize data
    input_data = input_normalization(input_data)
    # output_data = output_normalization(output_data)

    # Learning parameters
    lr = 0.1
    num_epochs = 25

    weight_matrix, MSE = cmac.train(input_data=input_data, output_data=output_data, num_epochs=num_epochs, alpha=lr, plot_table=False)

    # save cmac training
    np.save('/home/bio/bioinspired_ws/src/tutorial_4/data/cmac_weight_matrix.npy', np.asarray(weight_matrix))

    # Plot MSE for all epochs
    plt.plot(range(1, num_epochs+1), MSE)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.ylim(0, 1.1 * max(MSE))
    plt.title("Mean squared error of training at each epoch\n{} samples and CMAC with receptive field {}".format(num_samples, receptive_field_size))
    plt.show()

    '''
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
            _, output = cmac.predict(input_data=inputs_normalized, weight_matrix=weight_matrix)
            # Denormalize output
            #output = output_denormalization(output)
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
    '''
