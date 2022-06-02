#!/usr/bin/env python
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

class Central:


    def __init__(self):

        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.stiffness = False  
        self.blob_coordinates = []

        # define range of red color in HSV
        self.lower_red = np.array([161,155,84])
        self.upper_red = np.array([179,255,255])

        # define joint limits
        self.l_shoulder_pitch_limits = [rospy.get_param("joint_limits/left_shoulder/pitch")]
        self.l_shoulder_roll_limits = [rospy.get_param("joint_limits/left_shoulder/roll")]

        # define cmac parameters
        self.cmac_nb_inputs = 2
        self.cmac_nb_outputs = 2
        self.cmac_res = 50 # resolution
        self.cmac_field_size = 3 # receptive field size: set to 3 or 5 depending on the task, see tutorial description
        self.cmac_nb_neurons = self.cmac_field_size # to be defined, max field_size x field_size

        # receptive field: see additional material 3 on moodle. Here 5 neurons with coordinates in shape of a cross.
        #self.cmac_rf = [[0, 3], [1, 0], [2, 2], [3, 4], [4, 1]] 
        self.cmac_rf = [[0, 0], [1, 1], [2, 2]] 
        self.cmac_weight_table = np.random.uniform(-0.2, 0.2, (self.cmac_res, self.cmac_res, self.cmac_nb_outputs)) # Not all entries correspond to a neuron, depends on self.cmac_nb_neurons

        # define camera resolution
        # pixels idx run from 0 to resolution - 1
        self.cam_y_max = 240 - 1
        self.cam_x_max = 320 - 1

        # define training dataset
        self.path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_today.npy'
        self.training_dataset = np.load(self.path_to_dataset)
        self.nb_training_datapoints = 75 # set to 75 or 150 depending on the task, see tutorial description

        # create topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)

        # create topic subscribers
        self.redBlobPub = rospy.Publisher("red_blob_coordinates", Point, queue_size=1)  # Topic to publish the centroid coordinates of the red blob
        self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity


    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False


    def touch_cb(self,data):
        # Check which head button has been pressed. To trigger events on a button praise, raise
        # the corresponding flags.

        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))


    def image_cb(self,data):

        # declare instance of CVBrige for image conversion
        bridge_instance = CvBridge()

        # try convert the image
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")

        except CvBridgeError as e:
            rospy.logerr(e)
        
        # try detect a red blob
        try:
            self.cv2_blob_detection(cv_image)

        except:
            pass

        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    def cv2_blob_detection(self,image):

        # Transform image into HSV, select parts within the predefined red range color as a mask,
        # dilate and erode the selected parts to remove noise
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        kernel = np.ones((5,5),np.uint8)
        mask_dilation = cv2.dilate(mask, kernel, iterations=2)
        mask_final = cv2.erode(mask_dilation, kernel, iterations=1)
        kernel = np.ones((6,6),np.float32)/25
        mask_final = cv2.filter2D(mask_final,-1,kernel)

        # Apply mask to original image, show results
        res = cv2.bitwise_and(image,image, mask= mask_final)

        # Parameter definition for SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea  = True
        params.minArea = 1000
        params.maxArea = 200000
        params.filterByInertia = True
        params.minInertiaRatio = 0.0
        params.maxInertiaRatio  = 0.8

        # Applying the params
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(~mask_final)

        #draw 
        im_with_keypoints = cv2.drawKeypoints(~mask_final, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("Keypoints", im_with_keypoints)

        ## Find outer contours 
        im, contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maxContour = 0
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContour:
                maxContour = contourSize
                maxContourData = contour
               
        ## Draw
        cv2.drawContours(image, maxContourData, -1, (0,255,0), 2, lineType = cv2.LINE_4)

        # Calculate image moments of the detected contour
        M = cv2.moments(maxContourData)

        try:
        # Draw a circle based centered at centroid coordinates
            cv2.circle(image, (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), 5, (0, 0, 0), -1)

        # Show image:
            cv2.imshow("outline contour & centroid", image)

        except ZeroDivisionError:
            pass

        # Save center coordinates of the blob as a Point() message
        point_x = int(M['m10'] / M['m00'])
        point_y = int(M['m01'] / M['m00'])
        blob_coordinates_msg = Point(point_x, point_y, 0)

        # if the coordinates of a blob could be resolved
        if point_x != 0 and point_y != 0:
            # save in class variable
            self.blob_coordinates = [point_x, point_y]
        else:
            # keep the arm steady where it was before that moment
            pass

        # Publish center coordinates
        self.redBlobPub.publish(blob_coordinates_msg)
        

    # sets the stiffness for all joints.
    def set_stiffness(self,value):
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)

    
    def set_stiffness_tracking(self):
        #####
        # Stiffen up all joints and degrees of freedom but the shoulder pitch and roll
        # Outputs:
        #   Publication of JointState message on corresponding topic 
        #####

        # identify all joints of the joint state message
        joint_names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
                "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
                "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
                "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]
        position = [0.05058002471923828, -0.37126994132995605, 0.9433679580688477, 0.2039799690246582, -0.06600403785705566, -0.0367741584777832, -0.03992605209350586, 0.6187999844551086, -0.3481760025024414, -0.08432793617248535, -1.535889744758606, -0.09232791513204575, 0.9225810170173645, 0.029187917709350586, -0.3481760025024414, -0.06438612937927246, -1.535889744758606, -0.09232791513204575, 0.9226999878883362, 0.0, 0.9818019866943359, -0.29917192459106445, 0.8160459995269775, 0.3758718967437744, 1.3851600885391235, 0.3360000252723694]
        velocity = [0.1, 0.1, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        
        # select which joints are to be stiffened up. Command 1.0 in this case.
        effort =   [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # create JointState message
        stiffness_msg = JointState(Header(), joint_names, position, velocity, effort)

        # publish the message
        self.jointStiffnessPub.publish(stiffness_msg)
        rospy.sleep(1.0)

    
    def input_norm(self, coordinates):
        #####
        # this method takes as input a python list 1x2 corresponding to (x,y) coordinates and normalizes it
        # Inputs:
        #  coordinates: array of dim 2 x 1 containing the (x, y) pixel coordinates
        # Outputs:
        #  array of dim 2 x 1 containing the (x, y) pixel coordinates normalized to the iamge resolution
        #####

        return [float(coordinates[0]) / self.cam_x_max, float(coordinates[1]) / self.cam_y_max]


    def get_L2_neuron_position(self, input_data):
        #####
        # this method returns the position of neurons in L2 activated by a given input
        # Inputs:
        #  input_data: array of dim 2 x 1 containing the (x, y) normalized pixel coordinates
        # Outputs:
        #  neuron_pos: array of dimensions set by the size of the receptive field and number of ouputs
        #       containing the indices of the activated L2 neurons in the weight table
        #####

        # initialize variables
        position = [] # position of single L2 neuron
        neuron_pos = [] # positions of all L2 neurons
        displacement_list = [] # list of displacements along input dimensions
        quantized_ip_list = [] # list of quantized inputs
        
        # Perform quantization step (L1)
        for i_channel in range(self.cmac_nb_inputs):

            # quantize the input per the chosen resolution
            quantized_ip = int(self.input_norm(input_data)[i_channel] * self.cmac_res)

            # safety check to force the quantization to remain within boundaries
            if quantized_ip >= self.cmac_res:
                quantized_ip = self.cmac_res

            # append to the list
            quantized_ip_list.append(quantized_ip)

        # find coordinates of all activated L2 neurons
        for i_neuron in range(self.cmac_nb_neurons):
            
            # for all dimensions
            for inputs in range(self.cmac_nb_inputs):

                # compute the shift
                shift_amount  = (self.cmac_field_size - quantized_ip_list[inputs]) % self.cmac_field_size

                # compute local coordinates in receptive field
                local_coord = (shift_amount  + self.cmac_rf[i_neuron][inputs]) % self.cmac_field_size

                # compute L2 neuron coordinates in the weight tables
                coord = quantized_ip_list[inputs] + local_coord
                
                # append to list
                neuron_pos.append(coord) # why do we use a flat array for a set of (x,y) coordinates ? 
                # this can work but can also be misleading

        return neuron_pos


    def get_cmac_output(self, neuron_pos):
        # Calculate the ouput of the CMAC after L3
        # Inputs:
        #   neuron_pos: list of indices of the neurons within the receptive field, computed in L2. Can be viewed as activtion address vector
        # Outputs:
        #   x: list of values, each corresponding to an output of the CMAC

        # Initialize the outputs to 0
        x = [0] * self.cmac_nb_outputs

        # Loop through L3 neurons within the window (receptive field) selected in L2
        for jk_neuron in range(self.cmac_nb_neurons):

            # Loop through outputs
            for i_output in range(self.cmac_nb_outputs):

                row = neuron_pos[jk_neuron] # isn`t there an error here and it should be "2 * jk_neuron" ? not sure I follow the data structure
                col = neuron_pos[self.cmac_nb_neurons + jk_neuron] # isn`t there an error here and it should be "jk_neuron + 1" ?

                # Add weight from weight table
                x[i_output] = x[i_output] + self.cmac_weight_table[row, col, i_output]

        return x

    def train_cmac(self, cmac_weight_table, data, num_epochs):
        # Train the CMAC
        # Inputs:
        #   cmac_weight_table: untrained weight table
        #   data: training data of dim N samples x 4 where each sample is a (x, y) pixel coordinates 
        #       followed by (shoulder_pitch, shoulder_roll) joint state
        #   num_epochs: number of epochs for the training
        # Outputs:
        #   new_cmac_weight_table: trained weight table
        # Example call:
        #   self.cmac_weight_table = self.train_cmac(self.cmac_weight_table, self.training_dataset[0:149,:], 10)

        # Initialize variables
        new_cmac_weight_table = np.zeros(cmac_weight_table.shape) # Trained weight table
        alpha = 0.3 # Learning rate
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

            # Iterate through all data samples
            for d in range(len(data)):

                # Forward pass
                neuron_pos = self.get_L2_neuron_position(inputs[d, :])
                x = self.get_cmac_output(neuron_pos)

                # Compute MSE of data sample
                MSE_sample[d, epoch] = np.square(np.subtract(t[d],x)).mean()

                # Loop through L3 neurons within the window (receptive field) selected in L2
                for jk_neuron in range(self.cmac_nb_neurons):

                    # Loop through outputs
                    for i_output in range(self.cmac_nb_outputs):
                        row = neuron_pos[jk_neuron]
                        col = neuron_pos[self.cmac_nb_neurons + jk_neuron]
                        wijk = cmac_weight_table[row, col, i_output] # Weight to be updated
                        increment = alpha * (t[d, i_output] - x[i_output]) # Increment to be added
                        new_cmac_weight_table[row, col, i_output] = wijk + increment # New weight

            # Update weights for this epoch
            new_cmac_weight_table = cmac_weight_table

            # Print MSE of this epoch
            MSE[epoch] = MSE_sample[:, epoch].mean()
            print("MSE: " + str(MSE[epoch]))

        # Plot MSE across all epochs
        plt.plot(range(1, num_epochs+1), MSE)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Mean-Squared Error, Epoch: " + str(epoch + 1))
        plt.show()

        return new_cmac_weight_table


    # def get_L2_neuron_position(self):
    #     #####
    #     # this method returns the position of neurons in L2 activated by a given input
    #     #####

    #     position = []

    #     for i_neuron in range(self.cmac_nb_neurons):

    #         print("******************************")
    #         print("neuron # :", i_neuron)
    #         print("******************************")
    #         neuron_coord = []

    #         for i_channel in range(self.cmac_nb_inputs):

    #             print("*******")
    #             print("channel # :", i_channel)
                    
    #             input_index_q = int(self.input_norm(self.blob_coordinates)[i_channel] * self.cmac_res)
    #             print("shift idx :", input_index_q)

    #             shift_amount_d = self.cmac_field_size - input_index_q % self.cmac_field_size
    #             print("shift amount :", shift_amount_d)

    #             print("neuron coordinates in rf:", self.cmac_rf[i_neuron][i_channel])
    #             local_coord_p = (shift_amount_d + self.cmac_rf[i_neuron][i_channel]) % self.cmac_field_size
    #             print("local coordinates :", local_coord_p)

    #             coord = input_index_q + local_coord_p

    #             neuron_coord.append(coord)
    #             rospy.sleep(0.5)

    #         position.append(neuron_coord)
    #     print('******')
    #     print(len(position))
    #     print(position)



    #         #for i in range(len(position)):
    #          #   self.cmac_field_size

    #     return position

    def set_joint_angles(self,head_angle,joint_name):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)


    def central_execute(self):
        #####
        # main method of the node
        #####

        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10) # Allow joint control

        # display training dataset
        print("training data set is :", self.training_dataset)

        # set joint stiffness and ready for tracking task
        self.set_stiffness_tracking()
        print("*********")
        print("setting joints in stiff mode for tracking")
        print("setting joints in stiff mode")

        # Train the CMAC network
        num_data_samples = 150
        num_epochs = 10
        self.cmac_weight_table = self.train_cmac(self.cmac_weight_table, self.training_dataset[0:num_data_samples-1,:], num_epochs)

        while not rospy.is_shutdown():
            
            # self.set_stiffness_tracking() # Only for gathering training data!
            # self.set_stiffness(True)

            # if we have captured the coordinates of a blob
            if len(self.blob_coordinates) > 0:

                # compute which L2 neurons are activated
                neuron_pos = self.get_L2_neuron_position(self.blob_coordinates) # think we forgot to normalize the inputs here, see self.input_norm

                # compute the joint states
                x = self.get_cmac_output(neuron_pos)

                # publish and set the joint states
                self.set_joint_angles(x[0], "LShoulderPitch")
                self.set_joint_angles(x[1], "LShoulderRoll")

            rospy.sleep(1.0)

        # remove stiffness when the node stops executing
        self.set_stiffness(self.stiffness)


if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
