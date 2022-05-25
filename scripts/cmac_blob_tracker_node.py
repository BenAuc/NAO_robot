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

        # define cmac parameters
        self.cmac_nb_inputs = 2
        self.cmac_nb_outputs = 2
        self.cmac_res = 50 # resolution
        self.cmac_field_size = 3 # receptive field size: set to 3 or 5 depending on the task, see tutorial description
        self.cmac_nb_neurons = 5 # to be defined, max field_size x field_size
        # receptive field: see additional material 3 on moodle. Here 5 neurons with coordinates in shape of a cross.
        self.cmac_rf = [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]] 

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
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        #print('original image disp')
        #cv2.imshow("image window",cv_image)
        #print('running blob detec')
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
        #cv2.imshow('mask',mask_final)
        #cv2.imshow('image seen through mask',res)

        # Parameter definition for SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea  = True
        params.minArea = 1000
        params.maxArea = 200000
        params.filterByInertia = True
        params.minInertiaRatio = 0.0
        params.maxInertiaRatio  = 0.8

        #params.filterByConvexity = True
        #params.minConvexity = 0.09
        #params.maxConvexity = 0.99

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
        #cv2.imshow('image with countours',image)

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
        # self.blob_coordinates = [point_x, point_y]

        # if the coordinates of a blob could be resolved
        if point_x != 0 and point_y != 0:
            # save in class variable
            self.blob_coordinates = [point_x, point_y]
        else:
            # keep the arm steady where it was before that moment
            pass

        # Publish center coordinates
        self.redBlobPub.publish(blob_coordinates_msg)
        

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
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
        # this method stiffens up all joints and dfs but the shoulder to get ready to track the object
        #####

        joint_names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
                "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
                "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
                "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]

        position = [0.05058002471923828, -0.37126994132995605, 0.9433679580688477, 0.2039799690246582, -0.06600403785705566, -0.0367741584777832, -0.03992605209350586, 0.6187999844551086, -0.3481760025024414, -0.08432793617248535, -1.535889744758606, -0.09232791513204575, 0.9225810170173645, 0.029187917709350586, -0.3481760025024414, -0.06438612937927246, -1.535889744758606, -0.09232791513204575, 0.9226999878883362, 0.0, 0.9818019866943359, -0.29917192459106445, 0.8160459995269775, 0.3758718967437744, 1.3851600885391235, 0.3360000252723694]
        velocity = [0.1, 0.1, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        effort =   [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # stiffness_msg = JointState(Header(), self.stiff_joint_set, position, velocity, effort)
        stiffness_msg = JointState(Header(), joint_names, position, velocity, effort)

        self.jointStiffnessPub.publish(stiffness_msg)
        rospy.sleep(1.0)

    
    def input_norm(self, coordinates):
        #####
        # this method takes as input a python list 1x2 corresponding to (x,y) coordinates and normalizes it
        #####
        print("input x coor :", coordinates[0])
        print("input y coord:", coordinates[1])
        # print("max x :", self.cam_x_max)
        # print("max y :", self.cam_y_max)
        return [float(coordinates[0]) / self.cam_x_max, float(coordinates[1]) / self.cam_y_max]


    def get_L2_neuron_position(self):
        #####
        # this method returns the position of neurons in L2 activated by a given input
        #####

        position = []

        for i_neuron in range(self.cmac_nb_neurons):

            print("******************************")
            print("neuron # :", i_neuron)
            print("******************************")
            neuron_coord = []

            for i_channel in range(self.cmac_nb_inputs):

                print("*******")
                print("channel # :", i_channel)

                input_index = int(self.input_norm(self.blob_coordinates)[i_channel] * self.cmac_res)
                print("shift idx :", input_index)

                shift_amount = self.cmac_field_size - input_index % self.cmac_field_size
                print("shift amount :", shift_amount)

                print("neuron coordinates in rf:", self.cmac_rf[i_neuron][i_channel])
                local_coord = (shift_amount + self.cmac_rf[i_neuron][i_channel]) % self.cmac_field_size
                print("local coordinates :", local_coord)

                coord = input_index + local_coord

                neuron_coord.append(coord)
                rospy.sleep(0.5)

            position.append(neuron_coord)

        return position


    def central_execute(self):
        #####
        # main method of the node
        #####

        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # display training dataset
        print("training data set is :", self.training_dataset)

        # set joint stiffness and ready for tracking task
        self.set_stiffness_tracking()
        print("*********")
        print("setting joints in stiff mode for tracking")
        print("setting joints in stiff mode")

        while not rospy.is_shutdown():
            
            self.set_stiffness_tracking()

            if len(self.blob_coordinates) > 0:

                position = self.get_L2_neuron_position()

            rospy.sleep(1.0)

        # remove stiffness when the node stops executing
        self.set_stiffness(self.stiffness)


if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
