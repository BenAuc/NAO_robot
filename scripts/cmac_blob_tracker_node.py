#################################################################
# file name: cmac_training_script.py
# author's name: Diego
# created on: 30-05-2022
# last edit: 06-06-2022 (Benoit Auclair)
# function: ROS node that computes motor commands of shoulder joint 
#       to track a red blob in the visual field
#################################################################
# 
# #!/usr/bin/env python

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
        self.l_shoulder_pitch_limits = rospy.get_param("joint_limits/left_shoulder/pitch")
        self.l_shoulder_roll_limits = rospy.get_param("joint_limits/left_shoulder/roll")
        self.joint_limit_safety_f = rospy.get_param("joint_limits/safety")[0]
       
        # define cmac parameters
        self.cmac_nb_inputs = 2
        self.cmac_nb_outputs = 2
        self.cmac_res = 50 # resolution
        self.cmac_field_size = 3 # receptive field size: set to 3 or 5 depending on the task, see tutorial description
        self.cmac_nb_neurons = self.cmac_field_size # to be defined, max field_size x field_size
        self.path_to_weight_matrix = '/home/bio/bioinspired_ws/src/tutorial_4/data/cmac_weight_matrix.npy'


        # receptive field: see additional material 3 on moodle. Here 5 neurons with coordinates in shape of a cross.
        #self.cmac_rf = [[0, 3], [1, 0], [2, 2], [3, 4], [4, 1]] 
        self.cmac_rf = [[0, 0], [1, 1], [2, 2]] 
        self.cmac_weight_table = np.random.uniform(-0.2, 0.2, (self.cmac_res, self.cmac_res, self.cmac_nb_outputs)) # Not all entries correspond to a neuron, depends on self.cmac_nb_neurons

        # define camera resolution
        # pixels idx run from 0 to resolution - 1
        self.cam_y_max = 240 - 1
        self.cam_x_max = 320 - 1

        # define training dataset
        # self.path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_today.npy'
        self.path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_v1.npy'
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
        position = [-0.08748006820678711, -0.1764519214630127, 0.7899680137634277, 0.08432793617248535, -0.0353238582611084, -0.9602420330047607, -0.1503739356994629, 0.25679999589920044, -0.49390602111816406, 0.1104898452758789, -1.2701101303100586, -0.09232791513204575, 0.9225810170173645, -8.26716423034668e-05, -0.49390602111816406, -0.20551395416259766, -1.4235939979553223, 0.12582993507385254, 0.9226999878883362, 0.0, 1.8055601119995117, -0.26695799827575684, 1.8683700561523438, 0.581428050994873, -0.5200679302215576, 0.30879998207092285]
        velocity = [0.1, 0.1, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        
        # select which joints are to be stiffened up
        effort =   [0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        # create JointState message
        stiffness_msg = JointState(Header(), joint_names, position, velocity, effort)

        # publish the message
        self.jointStiffnessPub.publish(stiffness_msg)

        rospy.sleep(1.0)


        def set_joint_states_data_acquisition(self):
        #####
        # Set all joints in desired states prior to data acquisition
        # Outputs:
        #   Call to method that set the joint states 
        #####

            print("********")
            print("setting joint states in desired configuration")

            joint_names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
                    "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
                    "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
                    "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]

            position = [0.06, -0.22, 0.58, 0.24, -0.00, -0.91, -1.55, 0.26, -0.73, 0.18, -1.53, 0.64, 0.92, 0.0, -0.73, -0.18, -1.53, 0.81, 0.92, 0.0, 0.55, 0.0, 1.23, 0.36, -1.30, 0.30]

            self.set_stiffness_data_acquisition()

            for i_joint in range(len(joint_names)):
                self.set_joint_angles(position[i_joint], joint_names[i_joint])
                rospy.sleep(0.25)
                print("please wait...")

            print("all joint states have been configured -> ready for tracking")

    
    def input_normalization(self, coordinates):
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
        
        # NOTE: THIS LOOP WAS INTEGRATED IN THE FOLLOWING LOOP
        # # Perform quantization step (L1)
        # for i_channel in range(self.cmac_nb_inputs):

        #     # quantize the input per the chosen resolution
        #     quantized_ip = int(self.input_normalization(input_data)[i_channel] * self.cmac_res)

        #     # safety check to force the quantization to remain within boundaries
        #     if quantized_ip >= self.cmac_res:
        #         quantized_ip = self.cmac_res

        #     # append to the list
        #     quantized_ip_list.append(quantized_ip)

        # find coordinates of all activated L2 neurons
        for i_neuron in range(self.cmac_nb_neurons):
            position = []
            
            # for all dimensions
            for inputs in range(self.cmac_nb_inputs):

                # quantize the input per the chosen resolution
                quantized_ip = int(self.input_normalization(input_data)[inputs] * self.cmac_res)

                # safety check to force the quantization to remain within boundaries
                if quantized_ip >= self.cmac_res:
                    quantized_ip = self.cmac_res

                # compute the shift
                #
                # shift_amount  = (self.cmac_field_size - quantized_ip_list[inputs]) % self.cmac_field_size
                shift_amount  = (self.cmac_field_size - quantized_ip) % self.cmac_field_size

                # compute local coordinates in receptive field
                local_coord = (shift_amount  + self.cmac_rf[i_neuron][inputs]) % self.cmac_field_size

                # compute L2 neuron coordinates in the weight tables
                coord = quantized_ip + local_coord
                
                # append to list
                position.append(coord) # why do we use a flat array for a set of (x,y) coordinates ? 
                # this can work but can also be misleading

            # append to list
            neuron_pos.append(position)

        print("**************")
        print("set of L2 neurons activated :", neuron_pos)

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

                # fetch index of weight in table
                row = neuron_pos[jk_neuron][0]
                col = neuron_pos[jk_neuron][1] 

                # Add weight from weight table
                x[i_output] = x[i_output] + self.cmac_weight_table[row, col, i_output]

        # check whether joints are within their limit range and if not enforce it
        # check left shoulder pitch
        if x[0] > (1 - self.joint_limit_safety_f) * self.l_shoulder_pitch_limits[1]:
            print("Joint state pitch mapped out of its limits. Correction applied.", (1 - self.joint_limit_safety_f) * self.l_shoulder_pitch_limits[1])
            x[0] = (1 - self.joint_limit_safety_f) * self.l_shoulder_pitch_limits[1]

        elif x[0] < (1 - self.joint_limit_safety_f) * self.l_shoulder_pitch_limits[0]:
            print("Joint state pitch mapped out of its limits. Correction applied.", (1 - self.joint_limit_safety_f) * self.l_shoulder_pitch_limits[0])
            x[0] = (1 - self.joint_limit_safety_f) * self.l_shoulder_pitch_limits[0]

        # check left shoulder roll
        if x[1] > (1 - self.joint_limit_safety_f) * self.l_shoulder_roll_limits[1]:
            print("Joint state roll mapped out of its limits. Correction applied.", (1 - self.joint_limit_safety_f) * self.l_shoulder_roll_limits[1])
            x[1] = (1 - self.joint_limit_safety_f) * self.l_shoulder_roll_limits[1]

        elif x[1] < (1 - self.joint_limit_safety_f) * self.l_shoulder_roll_limits[0]:
            print("Joint state roll mapped out of its limits. Correction applied.", (1 - self.joint_limit_safety_f) * self.l_shoulder_roll_limits[0])
            x[1] = (1 - self.joint_limit_safety_f) * self.l_shoulder_roll_limits[0]

        return x


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
                    
    #             input_index_q = int(self.input_normalization(self.blob_coordinates)[i_channel] * self.cmac_res)
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

    #     print("**************")
    #     print("set of L2 neurons activated :", position)
        
    #     return position


    def set_joint_angles(self, head_angle, joint_name):

        print("call to : set_joint_angles()")

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

        # load trained cmac
        self.cmac_weight_table = np.load(self.path_to_weight_matrix)

        while not rospy.is_shutdown():
            
            #self.set_stiffness_tracking() # Only for gathering training data!

            # if we have captured the coordinates of a blob
            if len(self.blob_coordinates) > 0:

                # compute which L2 neurons are activated
                neuron_pos = self.get_L2_neuron_position(self.blob_coordinates)
                # neuron_pos = self.get_L2_neuron_position()

                # compute the joint states
                x = self.get_cmac_output(neuron_pos)

                # publish and set the joint states
                print("writing to shoulder pitch :", x[0])
                print("writing to shoulder roll :", x[1])

                self.set_stiffness(True)
                self.set_joint_angles(x[0], "LShoulderPitch")
                self.set_joint_angles(x[1], "LShoulderRoll")

            rospy.sleep(1.0)

        # remove stiffness when the node stops executing
        self.set_stiffness(self.stiffness)


if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
