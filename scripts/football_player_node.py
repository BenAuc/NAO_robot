#!/usr/bin/env python

# #################################################################
# file name: football_player_node.py
# author's name: Diego, Benoit, Priya, Vildana
# created on: 07-07-2022
# last edit: 07-07-2022 (Benoit Auclair)
# function: ROS node that computes the agent behavior (after training has been completed)
#################################################################

import rospy
from agent_environment_model import Agent, Policy
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from sensor_msgs.msg import Image, JointState

import numpy as np
import copy

class PenaltyKick:

    def __init__(self):

        # initialize class variables
        self.jointPub = None
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.stiffness = False  

        # define frequency of execution of this node
        self.frequency = 1 # Hz
        self.rate = rospy.Rate(self.frequency) # timing object

        # define joint limits
        self.r_hip_pitch_limits = rospy.get_param("joint_limits/right_hip/pitch")
        self.r_hip_roll_limits = rospy.get_param("joint_limits/right_hip/roll")
        print("hip joints : ", self.r_hip_pitch_limits, self.r_hip_roll_limits)

        self.r_ankle_pitch_limits = rospy.get_param("joint_limits/right_ankle/pitch")
        self.r_knee_pitch_limits = rospy.get_param("joint_limits/right_knee/pitch")
        print("hip joints : ", self.r_ankle_pitch_limits, self.r_knee_pitch_limits)

        self.joint_limit_safety_factor = rospy.get_param("joint_limits/safety")[0]

        # define resolution of the model
        # this parameter gets passed on to the model of the agent and of the environment
        # it sets the number of bins into which the range of the degrees of freedom are split
        self.resolution = 5

        # define where the learned policy is located
        # this policy is the output of another script which drove reinforcement learning
        self.learned_policy = '/home/bio/bioinspired_ws/src/tutorial_5/data/learned_policy.pickle'


        # define joint states and stand-up resting position
        # TODO what exactly the resting position ought to be
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


    def run(self):
        """
        Main loop of class.
        Inputs:
        -self
        Outputs:
        -runs the step function.
        """

        while not rospy.is_shutdown():

            # start computing behavior when the agent is in position
            if agent.readiness:

                # call to method to set Nao in upright position
                # TODO

                # perform step
                self.step()

                # sleep to target frequency
                self.rate.sleep()


    def step(self):
        """
        Perform an iteration of control.
        Inputs:
        -self
        Outputs:
        -sets the joint states.
        """
            
        ### to be completed ###
        pass


if __name__=='__main__':

    #initialize the node and set name
    rospy.init_node('markert_tracker',anonymous=True) #initilizes node

    # instantiate class and start loop function
    try:

        # instantiate main class of the node
        penalty_kick = PenaltyKick()

        # instantiate the agent
        agent = Agent(penalty_kick.resolution, penalty_kick.rest_position)

        # load the policy learned during training
        agent.load_policy(penalty_kick.learned_policy)

        # start the loop
        penalty_kick.run()
        
    except rospy.ROSInterruptException:
        pass




# Development notes: the following function can be very useful to control the NAO robot
"""
https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
https://github.com/immersive-command-system/Pose-Estimation-Aruco-Marker-Ros/blob/master/my_aruco_tracker/src/write_data.py
"""