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

        # define range of red color in HSV
        self.lower_red = np.array([161,155,84])
        self.upper_red = np.array([179,255,255])

        # acquisition of training data
        self.training_data = [] # variable containing the data points
        self.is_data_capture_on = False # flag to trigger recording of a data point
        self.is_list_saving_on = False # flag to trigger saving of the data points
        self.path_to_dataset = '/home/bio/bioinspired_ws/src/tutorial_4/data/training_data_today.npy' # path to store the training data

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

        # When button 1 is pressed, raise flag to capture a data point
        if data.button == 1 and data.state == 1:
            self.is_data_capture_on = True
            print("*********")
            print("button was hit -> capturing data point")
                
        # When button 2 is pressed, raise flag to save the list of acquired data points
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

        # # Parameter definition for SimpleBlobDetector
        # params = cv2.SimpleBlobDetector_Params()
        # params.filterByArea  = True
        # params.minArea = 1000
        # params.maxArea = 200000
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.0
        # params.maxInertiaRatio  = 0.8

        # # Applying the params
        # detector = cv2.SimpleBlobDetector_create(params)
        # keypoints = detector.detect(~mask_final)

        # #draw 
        # im_with_keypoints = cv2.drawKeypoints(~mask_final, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
            # save it in class variable
            self.blob_coordinates = [point_x, point_y]
        # otherwise
        else:
            # indicate that no blob has been identified
            self.blob_coordinates = []

        # publish center coordinates
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


    def record_data_point(self):
        #####
        # Capture one data point as array of dim 1 x 4 cointaining x, y pixel coordinates 
        #   and shoulder_pitch, shoulder_roll current joint states 
        # Outputs:
        #   Append data point to the class variable self.training_data 
        #####

        print("*********")
        print("call to method: record_data_point()")

        # if we have captured the coordinates of a blob
        if len(self.blob_coordinates) > 0:

            # make a copy of the (x,y) centroid of the blob
            data_point = copy.deepcopy(self.blob_coordinates)

            # append to the previous array the shoulder pitch and roll joint states
            data_point.append(self.joint_angles[2])
            data_point.append(self.joint_angles[3])

            # save the data point in the class variable
            self.training_data.append(data_point)

        # if no blob has been identified, print an error message
        else:
            print("no luck, there is currently no blob that's been identified")
        
        # lower the flag that triggers the recording of a data point
        self.is_data_capture_on = False


    def save_data_points(self):
        #####
        # Save in a .npy file all data points captured so far
        # Outputs:
        #   File containing the class variable self.training_data 
        #####

        print("*********")
        print("call to method: save_data_points()")

        # save the data points in a file
        np.save(self.path_to_dataset, np.asarray(self.training_data))

        # lower the flag that triggers the saving of a all data points
        self.is_list_saving_on = False


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles", JointAnglesWithSpeed, queue_size=10)

        # to publish the centroid coordinates of the red blob
        self.redBlobPub = rospy.Publisher("red_blob_coordinates", Point, queue_size=1)  

        # to stiffen up all joints for data acquisition
        self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        # set joint stiffness to be ready for data acquisition
        self.set_stiffness_data_acquisition()
        print("*********")
        print("setting joints in stiff mode for training data acquisition")

        while not rospy.is_shutdown():
            self.set_stiffness_data_acquisition()

            # if the flag has been raised, record a data point
            if self.is_data_capture_on:
                self.record_data_point()

            # if the flag has been raised, save all data points
            if self.is_list_saving_on:
                self.save_data_points()

        # remove stiffness when the node stops executing
        self.set_stiffness(self.stiffness)


if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
