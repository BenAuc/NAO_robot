#!/usr/bin/env python

# #################################################################
# file name: shoulder_controller_node.py
# author's name: Diego, Benoit, Priya, Vildana
# created on: 23-06-2022
# last edit: 27-06-2022 (Benoit Auclair): edits to denormalization method
# function: ROS node that computes motor commands of shoulder joint 
# to follow a red blob in the visual field
#################################################################

import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, Bumper, HeadTouch
from sensor_msgs.msg import JointState
from ffnn_model import FFNN, Linear
import pickle
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
        self.joint_limit_safety_factor = rospy.get_param("joint_limits/safety")[0]

        print('pitch limits :', self.l_shoulder_pitch_limits)

        # load trained model
        self.load_weight_path = '/home/bio/bioinspired_ws/src/tutorial_4/data/model_weights_NAO/weight_matrix.pickle'
        
        with open(self.load_weight_path, 'rb') as handle:
            model_params =  pickle.load(handle)

        self.model_controller = FFNN(num_inputs=2, num_outputs=2, num_layers=2, hidden_layer_dim=8, activation_func=Linear(), load_model=True, model_params=model_params)

        # define parameters to denormalize the output space
        self.load_normalization_path = "/home/bio/bioinspired_ws/src/tutorial_4/data/model_weights_NAO/fnn_normalization.pickle" 

        with open(self.load_normalization_path, 'rb') as handle:
            self.output_space_normalization =  pickle.load(handle)

        self.max_pitch = self.output_space_normalization['max_pitch']
        self.min_pitch = self.output_space_normalization['min_pitch']
        self.max_roll = self.output_space_normalization['max_roll']
        self.min_roll = self.output_space_normalization['min_roll']

        print('max_pitch :', self.max_pitch)
        print('min_pitch :', self.min_pitch)

        # define resting position of all joints during data acquisition
        self.joint_names = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
                "LHand", "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll", "RHipYawPitch",
                "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll", "RShoulderPitch", "RShoulderRoll",
                "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"]
        self.rest_position = [0.1594, -0.0721, 0.5322, 0.2254, 0.36658406257629395, -0.9725141525268555, -0.6289820671081543, 0.2656, -0.724006175994873, 0.19485998153686523, -1.535889744758606, 0.9096200466156006, 0.9225810170173645, -8.26716423034668e-05, -0.724006175994873, -0.1978440284729004, -1.535889744758606, 0.9219760894775391, 0.9226999878883362, 0.0, 0.6811380386352539, -0.19179201126098633, 1.2179540395736694, 0.32218194007873535, -1.3116121292114258, 0.30159997940063477]
        self.velocity = [0.1, 0.1, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        self.effort =   [0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

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
        """
        Set all joints in desired states as during data acquisition
        Inputs: none
        Outputs:
        -Call to method that set the joint states 
        """

        print("********")
        print("setting joint states in desired configuration")

        # stiffen up all joints as required
        self.set_stiffness(True)
        #self.set_stiffness_data_acquisition()

        # go through all joints and set at desired position
        for i_joint in range(len(self.joint_names)):
            self.set_joint_angles(self.rest_position[i_joint], self.joint_names[i_joint])
            rospy.sleep(0.15)
            print("please wait...")

        print("all joint states have been configured -> ready for tracking")


    def set_stiffness_data_acquisition(self):
        """
        Stiffen up all joints and degrees of freedom but the shoulder pitch and roll
        Inputs: none
        Outputs:
        -Publication of JointState message on corresponding topic 
        """

        # create message to stiffness desired joints
        stiffness_msg = JointState(Header(), self.joint_names, self.rest_position, self.velocity, self.effort)

        # publish to topc
        self.jointStiffnessPub.publish(stiffness_msg)

        rospy.sleep(0.5)


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

        # denormalize the output with the statistics of the training set
        scale = np.array([self.max_pitch, self.max_roll], dtype=float) - np.array([self.min_pitch, self.min_roll], dtype=float)
        output_denorm = np.multiply(ffnn_output, scale) + np.array([self.min_pitch, self.min_roll], dtype=float)

        # safety check
        # enforce model output within joint states limits

        if output_denorm[0, 0] < self.l_shoulder_pitch_limits[0] * self.joint_limit_safety_factor:
            output_denorm[0] = self.l_shoulder_pitch_limits[0] * self.joint_limit_safety_factor
            print('pitch corrected to :', self.l_shoulder_pitch_limits[0] * self.joint_limit_safety_factor)

        elif output_denorm[0, 0] > self.l_shoulder_pitch_limits[1] * self.joint_limit_safety_factor:
            output_denorm[0] = self.l_shoulder_pitch_limits[1] * self.joint_limit_safety_factor
            print('pitch corrected to :', self.l_shoulder_pitch_limits[1] * self.joint_limit_safety_factor)

        if output_denorm[0, 1] < self.l_shoulder_roll_limits[0] * self.joint_limit_safety_factor:
            output_denorm[1] = self.l_shoulder_roll_limits[0] * self.joint_limit_safety_factor
            print('roll corrected to :', self.l_shoulder_roll_limits[0] * self.joint_limit_safety_factor)

        elif output_denorm[0, 1] > self.l_shoulder_roll_limits[1] * self.joint_limit_safety_factor:
            output_denorm[1] = self.l_shoulder_roll_limits[1] * self.joint_limit_safety_factor
            print('roll corrected to :', self.l_shoulder_roll_limits[1] * self.joint_limit_safety_factor)

        return output_denorm


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

        rospy.sleep(3)

        # set joint state as done during data acquisition
        self.set_joint_states_data_acquisition()

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
            

            # Compute FFNN forward pass & output
            print('************ iteration ***************')
            print("blob coordinates :", self.object_coordinates)

            x_norm = self.model_controller.forward(self.object_coordinates.reshape(1,2))
            print('x_norm :', x_norm)

            # Denormalize FFNN output
            x_denorm = self.output_denormalization(x_norm)

            # publish and set the joint states
            print("writing to shoulder pitch :", x_denorm[0,0])
            print("writing to shoulder roll :", x_denorm[0,1])
            print('x_denorm :', x_denorm)
            
            self.set_stiffness(True)

            self.set_joint_angles(x_denorm[0,0], "LShoulderPitch")
            rospy.sleep(0.2)
            self.set_joint_angles(x_denorm[0,1], "LShoulderRoll")
            rospy.sleep(0.2)
            self.set_joint_angles(-0.36,'LElbowYaw') 
            rospy.sleep(0.2)
            self.set_joint_angles(-0.81,'LElbowRoll')
            rospy.sleep(0.2)


if __name__=='__main__':

    #initialize the node and set name
    rospy.init_node('object_tracker',anonymous=True) #initilizes node

    # instantiate class and start loop function
    try:
        shoulder_controller = ShoulderController()
        shoulder_controller.run()
        
    except rospy.ROSInterruptException:
        pass
