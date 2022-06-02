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
        # this method stiffens up all joints and dfs but the shoulder to get ready to acquire data points
        #####

        # COMMENT THESE LINES OUT TO GET EXPECTED BEHAVIOR
        # sets the joint involved in the training data acquisition in a pre-defined pose 
        #self.set_stiffness(self.stiffness)
        #self.set_stiffness(True)

        joint_names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
                "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
                "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
                "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]

        # position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # position = [0.0, 0.0, 0.0, 0.0, 0.0]
        # velocity = [0.0, 0.0, 0.0, 0.0, 0.0]
        # effort = [0.0, 0.0, 0.0, 0.0, 0.0]

        position = [0.05058002471923828, -0.37126994132995605, 0.9433679580688477, 0.2039799690246582, -0.06600403785705566, -0.0367741584777832, -0.03992605209350586, 0.6187999844551086, -0.3481760025024414, -0.08432793617248535, -1.535889744758606, -0.09232791513204575, 0.9225810170173645, 0.029187917709350586, -0.3481760025024414, -0.06438612937927246, -1.535889744758606, -0.09232791513204575, 0.9226999878883362, 0.0, 0.9818019866943359, -0.29917192459106445, 0.8160459995269775, 0.3758718967437744, 1.3851600885391235, 0.3360000252723694]
        velocity = [0.1, 0.1, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        effort =   [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # stiffness_msg = JointState(Header(), self.stiff_joint_set, position, velocity, effort)
        stiffness_msg = JointState(Header(), joint_names, position, velocity, effort)

        self.jointStiffnessPub.publish(stiffness_msg)
        rospy.sleep(1.0)


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
        
        self.is_data_capture_on = False


    def save_data_points(self):
        #####
        # this method saves in a file all data points captured so far
        #####
        print("*********")
        print("call to method: save_data_points()")

        np.save(self.path_to_dataset, np.asarray(self.training_data2))

        # with open('training_data_today.npy') as f:
        #     np.save(f, self.training_data2)

        self.is_list_saving_on = False


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

        # set joint stiffness and ready for data acquisition
        self.set_stiffness_data_acquisition()
        print("*********")
        print("setting joints in stiff mode for training data acquisition")

        while not rospy.is_shutdown():
            self.set_stiffness_data_acquisition()

            # if flag has been raised, record a data point
            if self.is_data_capture_on:
                self.record_data_point()

            # if flag has been raised, save all data points
            if self.is_list_saving_on:
                self.save_data_points()
            
            #rospy.sleep(1.0)

        # remove stiffness when the node stops executing
        self.set_stiffness(self.stiffness)


if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
