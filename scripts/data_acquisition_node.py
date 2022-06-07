#!/usr/bin/env python

#################################################################
# file name: data_acquisition_node.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 06-06-2022
# last edit: 07-06-2022 (Benoit)
# function: records training data points and saves it in a file
# outputs: .npy file that contains the data points
#################################################################

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
        self.jointPub = 0
        self.stiffness = False  
        self.do_repetitions_left = False    # Flag to perform repetitive left arm motion
        self.do_repetitions_right = False   # Flag for right arm to mirror left arm motion
        self.going_home = False             # Flag to bring arms to safe position

        # define range of blue color in HSV
        self.lower_red = np.array([161,155,84])
        self.upper_red = np.array([179,255,255])

        # joint to stiffen up when acquiring training data
        self.stiff_joint_set = ["LElbowYaw", "LElbowRoll", "HeadYaw", "HeadPitch", "LWristYaw"]
        self.training_data = np.zeros((150, 4))
        self.training_data2 = []
        self.is_data_capture_on = False
        self.is_list_saving_on = False
        self.path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_today.npy'


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        # self.blob_coordinates = []

        pass

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

        # Raise flag to capture a data point
        if data.button == 1 and data.state == 1:
            self.is_data_capture_on = True
            print("*********")
            print("button was hit -> capturing data point")
            # self.record_data_point()  
                
        # Raise flag to save the list of acquired data points
        if data.button == 2 and data.state == 1:
            self.is_list_saving_on = True
            print("*********")
            print("button was hit -> saving list of data points")


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
            # indicate that no blob has been identified
            self.blob_coordinates = []

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


    def set_joint_angles(self,head_angle,joint_name):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)

    
    def set_stiffness_data_acquisition(self):
        #####
        # Stiffen up all joints and degrees of freedom but the shoulder pitch and roll
        # Outputs:
        #   Publication of JointState message on corresponding topic 
        #####

        joint_names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
                "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
                "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
                "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]

        position = [-0.08748006820678711, -0.1764519214630127, 0.7899680137634277, 0.08432793617248535, -0.0353238582611084, -0.9602420330047607, -0.1503739356994629, 0.25679999589920044, -0.49390602111816406, 0.1104898452758789, -1.2701101303100586, -0.09232791513204575, 0.9225810170173645, -8.26716423034668e-05, -0.49390602111816406, -0.20551395416259766, -1.4235939979553223, 0.12582993507385254, 0.9226999878883362, 0.0, 1.8055601119995117, -0.26695799827575684, 1.8683700561523438, 0.581428050994873, -0.5200679302215576, 0.30879998207092285]
        velocity = [0.1, 0.1, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        effort =   [0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        stiffness_msg = JointState(Header(), joint_names, position, velocity, effort)

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

        position = [0.2, -0.22, 0.58, 0.24, -0.50, -0.51, -1.55, 0.26, -0.73, 0.18, -1.53, 0.64, 0.92, 0.0, -0.73, -0.18, -1.53, 0.81, 0.92, 0.0, 0.55, 0.0, 1.23, 0.36, -1.30, 0.30]

        self.set_stiffness_data_acquisition()

        for i_joint in range(len(joint_names)):
            self.set_joint_angles(position[i_joint], joint_names[i_joint])
            rospy.sleep(0.25)
            print("please wait...")

        print("all joint states have been configured -> ready for data acquisition")


    def record_data_point(self):
        #####
        # this method captures one data point and saves it in the class variable training_data2
        #####
        print("*********")
        print("call to method: record_data_point()")

        # if we have captured the coordinates of a blob
        if len(self.blob_coordinates) > 0:
            print("left shoulder pitch is : ", self.joint_angles[2])
            print("left shoulder roll is : ", self.joint_angles[3])
            print("blob coordinates are : ", self.blob_coordinates)

            data_point = copy.deepcopy(self.blob_coordinates)
            data_point.append(self.joint_angles[2])
            data_point.append(self.joint_angles[3])

            print("your acquired data point is : ", data_point)

            # save the data point in the class variable
            self.training_data2.append(data_point)

        else:
            print("no luck, there is currently no blob that's been identified")


    def save_data_points(self):
        #####
        # Save in a .npy file all data points captured so far
        # Outputs:
        #   .npy file containing all training data points
        #####
        print("*********")
        print("call to method: save_data_points()")

        np.save(self.path_to_dataset, np.asarray(self.training_data2))


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        #rospy.Subscriber("red_blob_coordinates",Point,self.read_blob_coordinates)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)
        self.redBlobPub = rospy.Publisher("red_blob_coordinates", Point, queue_size=1)  # Topic to publish the centroid coordinates of the red blob
        self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        rospy.sleep(3.0)

        # set joint stiffness and ready for data acquisition
        self.set_joint_states_data_acquisition()

        while not rospy.is_shutdown():
            self.set_stiffness_data_acquisition()
            #self.set_stiffness(False)

            # if flag has been raised, record a data point
            if self.is_data_capture_on:
                self.record_data_point()
                self.is_data_capture_on = False

            # if flag has been raised, save all data points
            if self.is_list_saving_on:
                self.save_data_points()
                self.is_list_saving_on = False

        # remove stiffness when the node stops executing
        self.set_stiffness(self.stiffness)


if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
