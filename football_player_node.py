#!/usr/bin/env python

# #################################################################
# file name: football_player_node.py
# author's name: Diego, Benoit, Priya, Vildana
# created on: 07-07-2022
# last edit: 07-07-2022 (Benoit Auclair)
# function: ROS node that computes the agent behavior (after training has been completed)
#################################################################

from audioop import avg
import rospy
from agent_environment_model import Agent, Policy
from std_msgs.msg import String, Header
from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Point, PolygonStamped
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from sensor_msgs.msg import Image, JointState
import numpy as np
import copy
import traceback
from naoqi import ALProxy

class PenaltyKick:

    def __init__(self):

        # NAO's IP
        self.robotIP = "10.152.246.219"

        # define frequency of execution of this node
        self.frequency = 0.15 # Hz
        self.rate = rospy.Rate(self.frequency) # timing object

        # define resolution of the model
        # this parameter gets passed on to the model of the agent and of the environment
        # it sets the number of bins into which the range of the degrees of freedom are split
        self.HIP_JOINT_RESOLUTION, self.GOALKEEPER_RESOLUTION = 5, 5 # Resolution for the quantization of the leg displacement and goalkeeper x coordinate

        # define goal keeper parameters
        # initialize to -1 position (x,y) of goal keeper and goal edges
        self.goal_keeper_x_position = -1
        self.goal_left_post_x_position = -1
        self.goal_right_post_x_position = -1

        # aruco markers TODO: assign the real values
        self.goal_keeper_marker_id = 38 # id of aruco marker on goal keeper
        self.goal_left_post_marker_id = 34 # ids of aruco markers on either side of the goal
        self.goal_right_post_marker_id = 35 # ids of aruco markers on either side of the goal

        # define where the learned policy is located
        # this policy is the output of another script which performed reinforcement learning
        self.path_to_learned_policy = '/home/bio/bioinspired_ws/src/tutorial_5/data/learned_policy.pickle'

        # define joint limits
        use_physical_limits = False
        if use_physical_limits:
            print('Start')
            self.r_hip_pitch_limits = rospy.get_param("joint_limits/right_hip/pitch")
            print(self.r_hip_pitch_limits)
            self.r_hip_roll_limits = rospy.get_param("joint_limits/right_hip/roll")

            self.r_ankle_pitch_limits = rospy.get_param("joint_limits/right_ankle/pitch")
            self.r_knee_pitch_limits = rospy.get_param("joint_limits/right_knee/pitch")

            self.joint_limit_safety_factor = rospy.get_param("joint_limits/safety")[0]
        else:
            # Use empirical values for the joint limits
            self.r_hip_roll_limits = [-1, 1]
            self.r_knee_pitch_limits = [-1, 1]

        # define joints 
        self.joint_names_state_variable = ["RHipRoll"]        
        self.joint_ids_state_variable = 15 # index of this joint as published by the topic  
        self.hip_roll_start_position = np.random.rand() * (
            self.r_hip_roll_limits[1] - self.r_hip_roll_limits[0]) + self.r_hip_roll_limits[0] # random hip roll start position
 
        self.joint_names_rest = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", 
        "LElbowRoll", "LWristYaw", "LHand", "LHipYawPitch", "LHipRoll", 
        "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
        "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", 
        "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"] # all joint names

        self.joint_rest_position = [-0.01535, -0.1917, 1.51247, 0.8656, -1.1965, 
        -0.3895, 0.06898, 0.2983, -0.17023, 0.459751, 
        -0.09043, -0.090548, 0.090465, 0.00881, -0.17023, 
        self.hip_roll_start_position, 0.20034, -0.088930, 0.390548, 0.13043,
        1.53557, 0.27338, 1.18420, 0.38814, 0.11194, 0.30239] # joint states for start, upright position

        # joint and joint states involved during kicking
        self.joint_names_kick = ["RHipPitch"]
        self.joint_position_kick = [-0.55034]
        self.joint_position_kick_home = [0.20034]
        self.joint_position_kick_before = [0.44034]

        # Flags 
        self.train_mode = True # to differentiate between train and forward mode
        self.agentIsReady = True # to indicate Nao is ready to start moving
        self.in_resting_position = False # to indicate whether now is in upright position or not

        # create topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState, self.joints_cb)
        rospy.Subscriber("tactile_touch", HeadTouch, self.touch_cb)
        rospy.Subscriber("/nao_robot/markerlist", PolygonArray, self.aruco_marker_parser)  # the list of polygons describing the outline of a marker + the marker ID

        # create topic publishers
        self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=1) # Allow joint control

        # initialize class variables
        self.stiffness = False  
        self.joint_names = [] 
        self.joint_angles = []
        self.joint_velocities = []


    def kick_ball(self):
        """
        Trigger kicking motion.
        Inputs:
        -None
        Outputs:
        -None
        """

        print("kicking initiated *****")
        
        self.set_joint_position(self.joint_names_kick, self.joint_position_kick_before)
        rospy.sleep(0.5)

        self.set_joint_position(self.joint_names_kick, self.joint_position_kick, speed=1)
        rospy.sleep(1.5)

        self.set_joint_position(self.joint_names_kick, self.joint_position_kick_home)
        rospy.sleep(1.5)


    def set_whole_body_stiffness(self,value):
        """
        Stiffen up all joints
        Inputs:
        -value: boolean whether to stiffen up (True) or release (False) the joints
        Outputs:
        -None
        """
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)


    def set_joint_position(self, joint_names, joint_position,speed=0.1):
        """
        Set all joints in desired position.
        Inputs:
        -None
        Outputs:
        -Call to method set_joint_angles()
        """

        # stiffen up all joints
        self.set_stiffness(True)

        # go through all joints and set at desired position
        for i_joint in range(len(joint_names)):
            self.set_joint_angles(joint_position[i_joint], joint_names[i_joint], speed=0.1)
            rospy.sleep(0.25)

        return True


    def set_joint_angles(self,head_angle,joint_name,speed):
        """
        Set all joints in desired state.
        Inputs:
        -head_angle: list of joint angles in radians
        -joint_name: list of joint names
        Outputs:
        -Publish joint states
        """

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = speed # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)
       
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


    def touch_cb(self,data):
        # Check which head button has been pressed. To trigger events on a button praise, raise
        # the corresponding flags.

        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

            
    def aruco_marker_parser(self, data):
        """
        Handle incoming PolygonArray() message with positions of detected Aruco markers.
        Inputs:
        -data: list of PolygonStamped() messages
        Outputs:
        -saves the location of goal keeper (numpy array 1 x 2 (x, y)) 
        and 2 edges of goal (numpy array 2 x 2 (x, y)) in class variables
        """

        for idx in range(len(data.polygons)):

            polygon = data.polygons[idx]

            marker_id = int(polygon.header.frame_id)

            # if the aruco marker is that of the goal keeper
            if marker_id == self.goal_keeper_marker_id:
                
                x_coordinates = []

                # make average of x and y coordinates of the marker to find its center
                for point in polygon.polygon.points:

                    x_coordinates.append(point.x)

                # store these coordinates
                self.goal_keeper_x_position = int(np.mean(x_coordinates))

            # if the aruco marker is that of the goal edge
            if marker_id == self.goal_left_post_marker_id:
                
                x_coordinates = []

                # make average of x and y coordinates of the marker to find its center
                for point in polygon.polygon.points:

                    x_coordinates.append(point.x)

                # store these coordinates
                self.goal_left_post_x_position = int(np.mean(x_coordinates))

            # if the aruco marker is that of the other edge of the goal
            if marker_id == self.goal_right_post_marker_id:
                
                x_coordinates = []

                # make average of x and y coordinates of the marker to find its center
                for point in polygon.polygon.points:

                    x_coordinates.append(point.x)

                # store these coordinates
                self.goal_right_post_x_position = int(np.mean(x_coordinates))


    def run(self):
        """
        Main loop of class.
        Inputs:
        -self
        Outputs:
        -runs the step function.
        """

        while not rospy.is_shutdown():

            if self.agentIsReady:

                # set Nao in upright position
                ###############################
                # FOR TESTING PURPOSES UNCOMMENT
                # WARNING: HOLD NAO REAL TIGHT IN THE AIR WHEN IT MOVES TO UPRIGHT POSITION
                ###############################
                # if not self.in_resting_position:
                #     print("setting jionts in resting position")
                #     print("please wait...")  
                #     rospy.sleep(4)
                #     self.in_resting_position = self.set_joint_position(self.joint_names_rest, self.joint_rest_position)
                #     print("all joint states have been configured -> ready for kicking")
                #     rospy.sleep(1.5)
            
                # if goal keeper and posts of the goal have been detected
                if self.goal_keeper_x_position >= 0 and self.goal_left_post_x_position >= 0 and self.goal_right_post_x_position >= 0:

                    print("agent is being instantiated with following parameters")
                    print("*************")
                    print("self.HIP_JOINT_RESOLUTION, self.hip_roll_start_position, self.GOALKEEPER_RESOLUTION, self.goal_keeper_x_position, [self.goal_left_post_x_position, self.goal_right_post_x_position]")
                    print(self.HIP_JOINT_RESOLUTION, self.hip_roll_start_position, self.GOALKEEPER_RESOLUTION, 
                        self.goal_keeper_x_position, [self.goal_left_post_x_position, self.goal_right_post_x_position])

                    # instantiate the agent
                    agent = Agent(hip_joint_resolution = self.HIP_JOINT_RESOLUTION, 
                        hip_joint_start_position = self.hip_roll_start_position,
                        goalkeeper_resolution = self.GOALKEEPER_RESOLUTION, 
                        goalkeeper_x = self.goal_keeper_x_position,
                        goal_lims = [self.goal_left_post_x_position, self.goal_right_post_x_position],
                        r_hip_roll_limits = self.r_hip_roll_limits, 
                        r_knee_pitch_limits = self.r_knee_pitch_limits
                        )

                    # load the policy learned during training
                    if not self.train_mode:
                        agent.load_policy(penalty_kick.path_to_learned_policy)

                    # if the button was pushed to start
                    while self.agentIsReady:

                        # read state variable stade
                        current_hip_roll = self.joint_angles[self.joint_ids_state_variable]

                        # perform step
                        hip_roll, action_id, previous_state_id = agent.step(current_hip_roll)

                        # compute action
                        # either kick the ball
                        if action_id == 2:
                            self.kick_ball()

                        # or update state variable (hip roll)
                        else:
                            self.set_joint_position(self.joint_names_state_variable, hip_roll)
                            rospy.sleep(0.5)

                        ## If in train mode, train the agent
                        if self.train_mode:
                            agent.train(action_id, previous_state_id)

                        # sleep to target frequency
                        self.rate.sleep()


    def step(self, agent):
        """
        Perform an iteration of control.
        Inputs:
        -self
        Outputs:
        -sets the joint states.
        """
            
        ### to be completed ###

        # the agent does an iteration on its state-action space based on identified positions of goal and goal keepers
        agent.step()


if __name__=='__main__':

    #initialize the node and set name
    rospy.init_node('markert_tracker',anonymous=True) #initilizes node

    # instantiate class and start loop function
    try:

        # instantiate main class of the node
        penalty_kick = PenaltyKick()

        # start the loop
        print("***** beginning of execution *****")
        penalty_kick.run()
        
    except Exception:
        traceback.print_exc()


# Development notes: the following function can be very useful to control the NAO robot
"""
https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
https://github.com/immersive-command-system/Pose-Estimation-Aruco-Marker-Ros/blob/master/my_aruco_tracker/src/write_data.py
"""