#!/usr/bin/env python

# #################################################################
# file name: cmac_training_script.py
# author's name: Diego, Benoit, Priya, Vildana
# created on: 30-05-2022
# last edit: 06-06-2022 (Benoit Auclair)
# function: ROS node that computes motor commands of shoulder joint 
#       to track a red blob in the visual field
#################################################################

import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from cmac_model import CMAC_Model, input_normalization, output_normalization, output_denormalization

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

class Central:

    def __init__(self):

        # initialize class variables
        self.jointPub = None
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
       
    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        pass
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

        for i_joint in range(len(joint_names)):
            self.set_joint_angles(position[i_joint], joint_names[i_joint])
            rospy.sleep(0.25)
            print("please wait...")

        print("all joint states have been configured -> ready for tracking")


    def set_joint_angles(self, head_angle, joint_name):

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

        cmac = CMAC_Model(num_inputs=2, num_outputs=2, res=50, receptive_field_size=5)
       
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)

        # create topic subscribers
        self.redBlobPub = rospy.Publisher("red_blob_coordinates", Point, queue_size=1)  # Topic to publish the centroid coordinates of the red blob
        self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10) # Allow joint control



        # load trained cmac
        weight_matrix = np.load('/home/bio/bioinspired_ws/src/tutorial_4/data/cmac_weight_matrix.npy')

        self.set_joint_states_data_acquisition() # Make sure starting position is same than in training data
        

        while not rospy.is_shutdown():

            # if we have captured the coordinates of a blob
            if len(self.blob_coordinates) > 0:
                # Normalize input
                data = input_normalization(self.blob_coordinates)
                # Compute CMAC output
                _, x_norm = cmac.predict(input_data=data, weight_matrix=weight_matrix)

                # Denormalize CMAC output
                #x = output_denormalization(x_norm)
                x=x_norm
                # publish and set the joint states
                print("writing to shoulder pitch :", x[0])
                print("writing to shoulder roll :", x[1])

                self.set_stiffness(True)
                self.set_joint_angles(x[0], "LShoulderPitch")
                self.set_joint_angles(x[1], "LShoulderRoll")
                self.set_joint_angles(-1,'LElbowYaw')
                self.set_joint_angles(-1,'LElbowRoll')

            rospy.sleep(0.5)

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
