#!/usr/bin/env python

# #################################################################
# file name: football_player_node.py
# author's name: Diego, Benoit, Priya, Vildana
# created on: 07-07-2022
# last edit: 19-07-2022 (Benoit)
# function: ROS node that computes the agent behavior either during 
# training or after training
#################################################################

from audioop import avg
import rospy
from agent_environment_model import Agent
from std_msgs.msg import String, Header
from jsk_recognition_msgs.msg import PolygonArray
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, HeadTouch
from sensor_msgs.msg import Image, JointState
import numpy as np
import copy
import traceback
from naoqi import ALProxy
import matplotlib.pyplot as plt

class PenaltyKick:
    """
    Class implementing the ROS node that sends the motor command based on a model of the agent, environment, and policy.
    This class can be set in training mode or not, see list of flags given as parameters.
    @Inputs:
    -all parameters defined in __init__()
    @Outputs:
    -perform iterations on the agent's behavior
    """

    def __init__(self):

        # NAO's IP
        self.robotIP = "10.152.246.219"

        # define frequency of execution of this node
        self.frequency = 0.15 # Hz
        self.rate = rospy.Rate(self.frequency) # timing object

        # define resolution of the model
        # this parameter gets passed on to the model of the agent and of the environment
        # it sets the number of bins into which the range of the degrees of freedom are split
        self.HIP_JOINT_RESOLUTION, self.GOALKEEPER_RESOLUTION = 8, 3 # Resolution for the quantization of the leg displacement and goalkeeper x coordinate

        # define goal keeper parameters
        # initialize to -1 position (x,y) of goal keeper and goal edges
        self.goal_keeper_x_position = -1
        self.goal_left_post_x_position = -1
        self.goal_right_post_x_position = -1

        # aruco markers used during penalty kick trials
        self.goal_keeper_marker_id = 38 # id of aruco marker on goal keeper
        self.goal_left_post_marker_id = 1 # ids of aruco markers on either side of the goal
        self.goal_right_post_marker_id = 2 # ids of aruco markers on either side of the goal

        # define where the learned policy is located
        # this policy is the output of another script which performed reinforcement learning
        self.path_to_learned_policy = '/home/bio/bioinspired_ws/src/tutorial_5/data/learned_policy.pickle'

        # joint and joint states involved during kicking
        self.joint_names_kick = ["RKneePitch"]
        self.joint_position_kick = [-0.05]
        self.joint_position_kick_home = [0.4]
        self.joint_position_kick_before = [1.4]

        # joint 
        self.joint_name_hip_swaying = ["LHipYawPitch"]
        self.swaying_amplitude = 0.25
        self.joint_name_hip_swaying_left = [-0.17023 + self.swaying_amplitude]
        self.joint_name_hip_swaying_right = [-0.17023 - self.swaying_amplitude]
        self.joint_name_hip_swaying_home = [-0.17023]

        # define joint limits
        use_physical_limits = False
        if use_physical_limits:
            print('Start')
            self.joint_limit_safety_factor = rospy.get_param("joint_limits/safety")[0]

            self.r_hip_pitch_limits = self.joint_limit_safety_factor * np.array(rospy.get_param("joint_limits/right_hip/pitch"))
            self.r_hip_roll_limits = self.joint_limit_safety_factor * np.array(rospy.get_param("joint_limits/right_hip/roll"))

            self.r_ankle_pitch_limits = self.joint_limit_safety_factor * np.array(rospy.get_param("joint_limits/right_ankle/pitch"))
            self.r_knee_pitch_limits = self.joint_limit_safety_factor * np.array(rospy.get_param("joint_limits/right_knee/pitch"))

        else:
            # Use empirical values for the joint limits
            self.joint_limit_safety_factor = rospy.get_param("joint_limits/safety")[0]
            self.r_hip_roll_limits = [-0.18, 0.05] # np.multiply([1, 0.5], self.joint_limit_safety_factor * np.array(rospy.get_param("joint_limits/right_hip/roll"))) # adjust the empirical limits based on actual joint limits
            self.r_knee_pitch_limits = [-1.2, 0.4]
            self.hip_yaw_pitch_limits = [-0.17023 - self.swaying_amplitude, -0.17023 + self.swaying_amplitude]

        # define joints with dof 
        self.joint_names_dof1_hiproll = ["RHipRoll"]
        self.joint_ids_state_variable = 15 # index of this joint as published by the topic 
        self.hip_roll_start_position = -0.18 # np.random.rand() * ( self.r_hip_roll_limits[1] - self.r_hip_roll_limits[0]) + self.r_hip_roll_limits[0] # random hip roll start position
            
        # self.joint_names_dof2_hipyawpitch  = ["LHipYawPitch"]
 
        self.joint_names_rest = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", 
        "LElbowRoll", "LWristYaw", "LHand", "LHipYawPitch", "LHipRoll", 
        "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
        "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", 
        "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"] # all joint names

        self.joint_rest_position = [0, 0, 1.51247, 1.0, -1.1965, 
        -0.3895, 0.06898, 0.2983, -0.17023, 0.05, 
        -0.09043, -0.090548, 0.090465, 0.18881, -0.17023, 
        self.hip_roll_start_position, 0, 0.4, 0.100548, 0.13043,
        1.53557, -0.6, 1.18420, 0.38814, 0.11194, 0.30239] # joint states for start, upright position

        # Flags 
        self.train_mode = True # to differentiate between train and forward mode
        self.agentIsReady = False # to indicate Nao is ready to start moving
        self.in_resting_position = False # to indicate whether now is in upright position or not
        self.toggle = False # used to prevent the flag from immediately toggling when button is released following a press
        self.agent_exist = False # to make sure only one agent is instatiated
        self.to_save = False # Falg to save the policy

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


    def run(self):
        """
        Main loop of class.
        Inputs:
        -self
        Outputs:
        -runs the step function.
        """

        while not rospy.is_shutdown():
            
            #print("hip roll limits set to :", self.r_hip_roll_limits)
            # if the agent is set to ready (push of button 1)
            if self.agentIsReady:

                # set Nao in upright position
                ###############################
                # FOR TESTING PURPOSES UNCOMMENT UNCOMMENT THE NEXT IF BLOCK
                # WARNING: HOLD NAO REAL TIGHT IN THE AIR WHEN IT MOVES TO UPRIGHT POSITION
                ###############################

                if not self.in_resting_position:
                    print("setting joints in upright position")
                    print("please wait...")  
                    self.set_joint_position(self.joint_names_rest, self.joint_rest_position)
                    self.in_resting_position = True
                    print("all joint states have been configured -> ready for kicking")
                    rospy.sleep(1)
            
                # testing of hip sway motion for second state variable
                # self.sway_hips()
                # rospy.sleep(1)

                ###############################
                # FOR TESTING PURPOSES UNCOMMENT THE NEXT IF BLOCK
                ###############################
                # if goal keeper and posts of the goal have been detected UNCOMMENT IF NECESSARY
                # print(self.goal_keeper_x_position, self.goal_left_post_x_position, self.goal_right_post_x_position)
                if self.goal_keeper_x_position >= 0 and self.goal_left_post_x_position >= 0 and self.goal_right_post_x_position >= 0:
                
                #if True:

                    # if no agent has been instantiated so far
                    if not self.agent_exist:

                        print("*************")
                        print("agent is being instantiated with following parameters")
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

                        # register the creation of the agent
                        self.agent_exist = True

                    # COMMENT IF NECESSARY
                    rospy.sleep(0.5)

                    train_mode_answer = raw_input("Do you want to train the robot? [y/n]: ")
                    if train_mode_answer == 'y':
                        self.train_mode = True
                        print('\n\nStarted in training mode...\n\n')
                    else:
                        self.train_mode = False
                        print('\n\nStarted in forward mode (not training)...\n\n')

                    print("*** New episode ***")

                    # load the policy learned during training
                    # agent.load_policy()
                    # print(agent.reward_per_episode)
                    # print(agent.policy)
                    if not self.train_mode:
                        agent.load_policy()
                    # as long as the agent is set to ready (push of button 1)
                    while self.agentIsReady:

                        self.episode(agent)

            
                        # sleep
                        rospy.sleep(0.2)
                        print("*** New episode ***")

                        agent.policy.print_explored_actions()
 


                        # saving the policy if button is pressed
                        agent.save_policy()

                        self.set_joint_position(self.joint_names_rest, self.joint_rest_position)

                    rospy.sleep(0.5)
            else:
                #self.set_stiffness(False)
                pass

    def episode(self, agent):
        """
        Perform an iteration of control.
        Inputs:
        -self
        Outputs:
        -sets the joint states.
        """

        agent.a1 = agent.environment.state_id_to_coords(agent.current_state_id)[0]
        agent.a2 = 0
            
        # read state variable stade
        current_hip_roll = self.joint_angles[self.joint_ids_state_variable]
        print("current_hip_roll :", current_hip_roll)

        # the agent does an iteration on its state-action space
        while True:
            hip_roll, knee_pitch, action_id, previous_state_id = agent.step(current_hip_roll)
            print("Action ID: {}".format(action_id))
            
            if action_id == 2:
                self.kick_ball()

                ## If in train mode, train the agent
                if self.train_mode:
                    agent.train(action_id, previous_state_id)

                break
            else:
                print(self.joint_names_dof1_hiproll, hip_roll)
                self.set_joint_position(self.joint_names_dof1_hiproll, hip_roll)
                rospy.sleep(0.5)

                ## If in train mode, train the agent
                if self.train_mode:
                    agent.train(action_id, previous_state_id)
                



    def kick_ball(self):
        """
        Trigger kicking motion.
        Inputs:
        -None
        Outputs:
        -call to set_joint_position() to send motor commands to NAO
        """

        print("***** kicking initiated *****")
        
        # move backward to get some swing
        self.set_joint_position(self.joint_names_kick, self.joint_position_kick_before)
        rospy.sleep(0.5)

        # send leg forward past the ball to kick it
        self.set_joint_position(self.joint_names_kick, self.joint_position_kick, speed=0.8)
        rospy.sleep(2)

        # brings the leg back to its home position
        self.set_joint_position(self.joint_names_kick, self.joint_position_kick_home)
        rospy.sleep(1)


    # def sway_hips(self):
    #     """
    #     Trigger hip swaying motion.
    #     Inputs:
    #     -None
    #     Outputs:
    #     -call to set_joint_position() to send motor commands to NAO
    #     """

    #     print("***** hip swaying initiated *****")
        
    #     # move backward to get some swing
    #     self.set_joint_position(self.joint_name_hip_swaying, self.joint_name_hip_swaying_left, speed=0.5)
    #     rospy.sleep(1.5)

    #     # send leg forward past the ball to kick it
    #     self.set_joint_position(self.joint_name_hip_swaying, self.joint_name_hip_swaying_right, speed=0.5)
    #     rospy.sleep(1.5)

    #     # brings the leg back to its home position
    #     self.set_joint_position(self.joint_name_hip_swaying, self.joint_name_hip_swaying_home, speed=10.5)
    #     rospy.sleep(1.5)


    def set_whole_body_stiffness(self, value):
        """
        Stiffen up all joints
        Inputs:
        -value: boolean whether to stiffen up (True) or release (False) the joints
        Outputs:
        -call to ROS service
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


    def set_joint_position(self, joint_names, joint_position,speed = 0.1):
        """
        Set all joints in desired upright position.
        Inputs:
        -joint_names: python array with the strings of joint names
        - joint_position: python array with the ints of joint states
        - speed: ints of joint speed
        Outputs:
        -Call to method set_joint_angles()
        """

        # stiffen up all joints
        self.set_stiffness(True)

        # go through all joints and set at desired position
        for i_joint in range(len(joint_names)):
            self.set_joint_angles(joint_position[i_joint], joint_names[i_joint], speed=speed)
            rospy.sleep(0.05)

        return


    def set_joint_angles(self,head_angle,joint_name,speed):
        """
        Set all joints in desired state.
        Inputs:
        -head_angle: list of joint angles in radians
        -joint_name: list of joint names
        -speed: list of speeds at which the command should be executed
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
        """
        Handle inconing joint state message.
        Inputs:
        -data: joint state message
        Outputs:
        -save to class variables
        """
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity


    # sets the stiffness for all joints.
    def set_stiffness(self,value):
        """
        Set whole body stiffness.
        Inputs:
        -value: boolean whether the body should be stiff or not stiff
        Outputs:
        -Publish joint states
        """
        if value == True:
            service_name = '/body_stiffness/enable'
        elif value == False:
            service_name = '/body_stiffness/disable'
            print("**tap button 1 on NAO's head so set him in upright position and get the loop iterations started **")
        try:
            stiffness_service = rospy.ServiceProxy(service_name,Empty)
            stiffness_service()
        except rospy.ServiceException, e:
            rospy.logerr(e)


    def touch_cb(self,data):
        """
        Handling incoming button state message and trigger action or raise corresponding flags.
        Inputs:
        -value: boolean whether the body should be stiff or not stiff
        Outputs:
        -Publish joint states
        """

        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

        # setting the readiness to true with a touch button
        if not self.agentIsReady and data.button == 1 and data.state == 1:
            print("***** agent is set to ready *****")
            self.agentIsReady = True
            self.toggle = True

        # setting the readiness to true with a touch button
        elif self.agentIsReady and data.button == 1 and data.state == 0:
            if self.toggle:
                self.toggle = False
            else:
                print("***** agent is set to not ready *****")
                self.agentIsReady = False
                self.in_resting_position = False
        
        # if data.button == 2 and data.state == 1:
        #     self.to_save = True
            

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


if __name__=='__main__':

    #initialize the node and set name
    rospy.init_node('FootballPlayerNode',anonymous=True) #initilizes node

    # instantiate class and start loop function
    try:

        # instantiate main class of the node
        penalty_kick = PenaltyKick()

        # start the loop
        print("***** football player node is running *****")
        print("**tap button 1 on NAO's head so set him in upright position and get the loop iterations started **")
        penalty_kick.run()
        
    except Exception:
        traceback.print_exc()


# Development notes: the following function can be very useful to control the NAO robot
"""
https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
https://github.com/immersive-command-system/Pose-Estimation-Aruco-Marker-Ros/blob/master/my_aruco_tracker/src/write_data.py
"""
