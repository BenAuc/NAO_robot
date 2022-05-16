#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False  
        self.do_repetitions_left = False
        self.do_repetitions_right = False
        self.going_home = False
        self.r_arm_touch = False

        # define range of blue color in HSV
        self.lower_red = np.array([161,155,84])
        self.upper_red = np.array([179,255,255])

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False

    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

        if data.button == 1 and data.state == 1:
            self.going_home = True
            self.do_repetitions_left = False
            self.do_repetitions_right = False
            print("*********")
            print("arms going home")

        #if data.button == 3 and not self.r_arm_touch:
            #self.do_repetitions_left = True

        if data.button == 2 and data.state == 1 and not self.do_repetitions_left:
            self.do_repetitions_left = True
            self.going_home = False
            print("*********")
            print("starting repetitive left arm motion routine")

        if data.button == 3 and data.state == 1 and self.do_repetitions_left: 
            self.do_repetitions_right = True
            print("*********")
            print("starting repetitive right arm motion routine")

        # else:
        #     if data.button == 2 and data.state == 1 and self.do_repetitions_left:
        #         self.do_repetitions_left = False
        #         print("stopping repetitive left arm motion routine")
        #         print("*********")

        #if data.button == 3 and not self.r_arm_touch:
            #self.do_repetitions_left = True

    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        #print('original image disp')
        cv2.imshow("image window",cv_image)
        #print('running blob detec')
        try:
            self.cv2_blob_detection(cv_image)

        except:
            pass

        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    def cv2_blob_detection(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        kernel = np.ones((5,5),np.uint8)
        mask_dilation = cv2.dilate(mask, kernel, iterations=2)
        mask_final = cv2.erode(mask_dilation, kernel, iterations=1)
        kernel = np.ones((6,6),np.float32)/25
        mask_final = cv2.filter2D(mask_final,-1,kernel)

        res = cv2.bitwise_and(image,image, mask= mask_final)
        cv2.imshow('mask',mask_final)
        cv2.imshow('image seen through mask',res)

        # Parameter def
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

        # Applying the param
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(~mask_final)

        #draw 
        im_with_keypoints = cv2.drawKeypoints(~mask_final, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)

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
        cv2.imshow('image with countours',image)

        # Calculate image moments of the detected contour
        M = cv2.moments(maxContourData)

        try:
        # Draw a circle based centered at centroid coordinates
            cv2.circle(image, (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), 5, (0, 0, 0), -1)

        # Show image:
            cv2.imshow("outline contour & centroid", image)

        except ZeroDivisionError:
            pass

        blob_coordinates_msg = Point(int(M['m10'] / M['m00']), int(M['m01'] / M['m00']), 0)
        # blob_coordinates_msg.x = int(M['m10'] / M['m00'])
        # blob_coordinates_msg.y = int(M['m01'] / M['m00'])
        # blob_coordinates_msg.z = 0

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
        
    def left_arm_home(self):

        self.set_stiffness(True) 
        
        self.set_joint_angles(1.8, "LShoulderPitch")
        self.set_joint_angles(0.4, "LShoulderRoll")

        self.set_joint_angles(-0.4, "LElbowYaw")
        self.set_joint_angles(-0.4, "LElbowRoll")

        rospy.sleep(3.0)

        #self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!


    def both_arms_home(self):

        self.set_stiffness(True) 
        
        self.set_joint_angles(1.8, "RShoulderPitch")
        self.set_joint_angles(-0.4, "RShoulderRoll")

        self.set_joint_angles(0.4, "RElbowYaw")
        self.set_joint_angles(0.4, "RElbowRoll")

        self.set_joint_angles(1.8, "LShoulderPitch")
        self.set_joint_angles(0.4, "LShoulderRoll")

        self.set_joint_angles(-0.4, "LElbowYaw")
        self.set_joint_angles(-0.4, "LElbowRoll")

        rospy.sleep(3.0)

        #self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!


    def repeat_move(self):

        while self.do_repetitions_left:
            print("iterating repetitive arm motion")

            #self.left_arm_home()
            
            self.set_stiffness(True)
            
            self.set_joint_angles(0.35, "LShoulderPitch")
            self.set_joint_angles(1.05, "LShoulderRoll")

            if self.do_repetitions_right:

                self.set_joint_angles(-0.35, "RShoulderPitch")
                self.set_joint_angles(-1.05, "RShoulderRoll")

            self.set_joint_angles(-1.4, "LElbowYaw")
            self.set_joint_angles(-1.4, "LElbowRoll")

            if self.do_repetitions_right:

                self.set_joint_angles(1.4, "RElbowYaw")
                self.set_joint_angles(1.4, "RElbowRoll")

            #if right_arm_mirror

            rospy.sleep(2.0)

            if self.do_repetitions_right:
                self.both_arms_home()
            else:
                self.left_arm_home()

        # self.left_arm_home()


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)
        self.redBlobPub = rospy.Publisher("red_blob_coordinates", Point, queue_size=1)


        # test sequence to demonstrate setting joint angles
        # self.set_stiffness(True) # don't let the robot stay enabled for too long, the motors will overheat!! (don't go for lunch or something)
        # rospy.sleep(1.0)
        # self.set_joint_angles(0.5, "RShoulderPitch")
        # rospy.sleep(3.0)
        # self.set_joint_angles(0.0, "RShoulderPitch")
        # rospy.sleep(3.0)
        # self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        while not rospy.is_shutdown():
            self.set_stiffness(self.stiffness)

            if self.do_repetitions_left:
                self.repeat_move()

            if self.going_home: 
                self.both_arms_home()
                self.going_home = False

        # if data.button == 3 and not self.r_arm_touch:
            
        #     self.left_arm_repeat_move()
        #     rate.sleep()
            

    #rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
