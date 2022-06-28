#!/usr/bin/env python

# #################################################################
# file name: shoulder_controller_node.py
# author's name: Diego, Benoit, Priya, Vildana
# created on: 23-06-2022
# last edit: 06-06-2022 (Benoit Auclair)
# function: ROS node that computes motor commands of shoulder joint 
# to follow a red blob in the visual field
#################################################################

import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from sensor_msgs.msg import Image, JointState

import numpy as np
import copy

class ShoulderController:

    def __init__(self):

        # initialize class variables
        self.jointPub = None
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.stiffness = False  
        self.object_coordinates = None

        # define frequency of execution of this node
        self.frequency = 1 # Hz
        self.rate = rospy.Rate(self.frequency) # timing object

        # define joint limits
        self.l_shoulder_pitch_limits = rospy.get_param("joint_limits/left_shoulder/pitch")
        self.l_shoulder_roll_limits = rospy.get_param("joint_limits/left_shoulder/roll")
        self.joint_limit_safety_f = rospy.get_param("joint_limits/safety")[0]

        # define output space of the nodes where the shoulder joint state is mapped on to
        self.max_pitch = 0.06
        self.min_pitch = -0.63
        self.max_roll = 0.39
        self.min_roll = -0.32

        # define joints and resting position
        self.joint_names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
                "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
                "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
                "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]
        self.rest_position = [0.06, -0.22, 0.58, 0.24, -0.00, -0.91, -1.55, 0.26, -0.73, 0.18, -1.53, 0.64, 0.92, 0.0, -0.73, 
        -0.18, -1.53, 0.81, 0.92, 0.0, 0.55, 0.0, 1.23, 0.36, -1.30, 0.30]

        # create topic subscribers
        self.object_tracker_sub = rospy.Subscriber("/nao_robot/tracked_object/coordinates", Point, self.object_tracking) # fetch coordinates of tracked object
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)

        # create topic publishers
        self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10) # Allow joint control

       
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

        for i_joint in range(len(self.joint_names)):
            self.set_joint_angles(self.rest_position[i_joint], self.joint_names[i_joint])
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


    def output_denormalization(self, ffnn_output):
        """
        Denormalize the output of the feed forward neural network
        Inputs:
        -ffnn_output: array of dimensions 2x1
        Outputs:
        -mapping to output space of shoulder joint
        """

        return ffnn_output  * (np.array([self.max_pitch, self.max_roll], dtype=float) - np.array([self.min_pitch, self.min_roll], dtype=float)) + np.array([self.min_pitch, self.min_roll], dtype=float)


    def object_tracking(self, data):
        """
        Handles incoming message containing the coordinates of the tracked object.
        Inputs:
        -dta: incoming Point() message
        Outputs:
        -saves the object coordinates as array of (x,y) coordinates in the class variable self.object_coordinates
        """
        # extract data from incoming message
        point_x = data.x
        point_y = data.y

        # if the message contains a coordinate that's not the origin
        if point_x > 0 and point_y > 0:
            self.object_coordinates = np.asarray([point_x, point_y])

        # if no coordinate was received assign None to the varibale to indicate that no object is tracked
        else: 
            self.object_coordinates = None


    def run(self):
        """
        Main loop of class.
        Inputs:
        -self
        Outputs:
        -runs the step function.
        """

        while not rospy.is_shutdown():
            # perform step
            self.step()
            # sleep to target frequency
            self.rate.sleep()


    def step(self):
        """
        Perform an iteration of shoulder control.
        Inputs:
        -self
        Outputs:
        -sets the joint states.
        """

        if self.object_coordinates is not None:
            
            ### to be completed ###
            pass


if __name__=='__main__':

    #initialize the node and set name
    rospy.init_node('markert_tracker',anonymous=True) #initilizes node

    # instantiate class and start loop function
    try:
        shoulder_controller = ShoulderController()
        shoulder_controller.run()
        
    except rospy.ROSInterruptException:
        pass




# Development notes: the following function can be very useful to control the NAO robot
"""
https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
https://github.com/immersive-command-system/Pose-Estimation-Aruco-Marker-Ros/blob/master/my_aruco_tracker/src/write_data.py
"""