#!/usr/bin/env python

#################################################################
# file name: agent_environment_model.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 30-06-2022
# last edit: 07-07-2022 
# function: define the Agent, Policy, and Environment class
#################################################################


from hmac import new
import rospy
from geometry_msgs.msg import Point, PolygonStamped
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, HeadTouch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt



class Agent:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, resolution, start_position):

        # variable containing the policy the agent has so far learned
        self.policy = Policy(4, resolution ** 2, low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll, resolution)

        # compute the linear id of the agent's current state in the policy table based on the 2 hip joint states given as start position vector
        self.current_state_id = np.ravel_multi_index((start_position[0], start_position[1])) 

        # define label easier to understand for each of the actions
        self.action_dictionary = {
            0: 'right',
            1: 'up',
            2: 'left',
            3: 'down'
        }

        # flag to indicate that the agent is in position and ready to kick a goal
        # TODO: we can couple this flat to the press of a button to allow us to trigger the node when we want it to execute actions
        self.readiness = False

        # create topic publishers
        self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10) # Allow joint control


    def step(self):
        """
        Perform step.
        Inputs:
        -...
        Outputs:
        -...
        """

        # request computation of the id of the action to take given the current state
        action_id = self.policy.select_next_action(self.current_state_id)

        # Gien the ids of the current state and the action, update the current state to the next state and check if it lies within environment boundaries
        self.current_state_id, is_in_bounds = self.policy.environment.move_to_next_state(self.current_state_id, action_id)

        # Raise an error if the computed state is not within the boundaries of the environment
        if not is_in_bounds:
            raise ValueError('The computed next state of the agent lies outside the boundaries of the environment.')


    def set_joint_angles(self, head_angle, joint_name):
        """
        Handles incoming motor command to set the joint state.
        Inputs:
        -joint_name: string containing the joint name
        -head_angle: float containing the desired state
        Outputs:
        -triggers the motion
        """

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)

        return joint_angles_to_set


    def load_policy(self,path):
        """
        Upload the policy learned during a prior training
        Inputs:
        -path: path to the .pickle file containing the policy
        Outputs:
        -self.policy: store the policy in the class variable
        """

        pass


    def save_policy(self,path):
        """
        Upload the policy learned during a prior training
        Inputs:
        -path: path to the .pickle file containing the policy
        Outputs:
        -self.policy: store the policy in the class variable
        """

        pass


class Policy:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, nb_actions, nb_states, low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll, resolution):
        
        # self.value_function = [] # array of size (resolution, resolution)
        
        #self.environment = Environment() # array of size (resolution, resolution)

        # define state action pair
        # reward: the  reward / return obtainable in that state
        # next_state: state transition when the action is selected
        # done: whether the state has been sufficiently visited
        # nb_visits: number of times the state has been visited
        self.state_action_pair = {'reward': 0, 'next_state': 0, 'visited': False, 'is_valid': True}

        self.nb_actions = nb_actions

        # define e-greedy policy epsilon parameter
        self.exploitation_threshold = 0.3 # threshold when the agent starts exploiting the knowledge of the environment rather than exploring

        self.resolution = resolution

        # variable containing the environment the policy is about
        # in this case this is the 2 degrees of freedom of the hip
        self.environment = Environment(low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll, resolution)

        self.policy = []

        # Fill the self.policy array (nb_states-times-nb_actions) with self.state_action_pair's
        for state_id in range(nb_states):
            # Create new row
            new_state = []
            # Loop through columns
            for action_id in range(nb_actions):
                # Store self.state_action_pair in the cell
                new_state.append(self.state_action_pair)

                # Check if the action is valid (action doesn't lead the agent outside the boundaries of the environment)
                _, next_state_is_in_bounds = self.environment.move_to_next_state(state_id, action_id)

                # Update the validity of the action according to next_state_is_in_bounds
                new_state[action_id]['is_valid'] = next_state_is_in_bounds

            # Append row to the list
            self.policy.append(new_state)

    def plot(self):
        """
        Plot the policy
        """

        V = np.zeros((self.resolution ** 2, 1))
        draw_vals = True
        plt.rcParams['figure.dpi'] = 175
        plt.rcParams.update({'axes.edgecolor': (0.32,0.36,0.38)})
        plt.rcParams.update({'font.size': 4})
        plt.figure(figsize=(self.resolution, self.resolution))
        plt.imshow(1 - V.reshape(self.resolution, self.resolution), cmap='gray', interpolation='none', clim=(0,1))
        ax = plt.gca()
        ax.set_xticks(np.arange(self.resolution)-.5)
        ax.set_yticks(np.arange(self.resolution)-.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for state_id in range(self.policy.shape[0]):
            x = state_id%self.resolution
            y = int(state_id/self.resolution)
            action = self.policy[state_id]
            gray = np.array((0.32,0.36,0.38))
            # Draw arrows
            if action[0] > 0.0: plt.arrow(x, y, float(action[0])*.84, 0.0,  color=gray+0.2*(1-V[state_id]), head_width=0.1, head_length=0.1) # right
            if action[1] > 0.0: plt.arrow(x, y, 0.0, float(action[1])*-.84, color=gray+0.2*(1-V[state_id]), head_width=0.1, head_length=0.1) # up
            if action[2] > 0.0: plt.arrow(x, y, float(action[2])*-.84, 0.0, color=gray+0.2*(1-V[state_id]), head_width=0.1, head_length=0.1) # left
            if action[3] > 0.0: plt.arrow(x, y, 0.0, float(action[3])*.84,  color=gray+0.2*(1-V[state_id]), head_width=0.1, head_length=0.1) # down

            if draw_vals and V[state_id]>0:
                vstr = '{0:.1e}'.format(V[state_id]) if self.resolution == 8 else '{0:.6f}'.format(V[state_id])
                plt.text(x-0.45,y+0.45, vstr, color=(gray*V[state_id]), fontname='OpenSans')
        plt.grid(color=(0.42,0.46,0.48), linestyle=':')
        ax.set_axisbelow(True)
        ax.tick_params(color=(0.42,0.46,0.48),which='both',top='off',left='off',right='off',bottom='off')
        plt.show()


    def select_next_action(self, state):
        """
        Handles ...
        Inputs:
        -state: int containing the id of the current state
        Outputs:
        - next_action_id: int containing the id of the action to take
        """

        # intialize a list of unvisited states
        unvisited_states = []

        # intialize a list of rewards of the state-action pairs
        rewards = []

        # check whether each of the actions has been taken
        for action in range(self.nb_actions):

            # store the reward at the state-pair
            rewards.append(self.policy[state][action]['reward'])

            # check if state-pair has been visited
            if (not self.policy[state][action]['visited']):
                unvisited_states.append(action)

        # generate random number
        greedy_probability = np.random.randn(1)

        # if larger than e-greedy threshold then aim for greatest reward
        if greedy_probability > self.exploitation_threshold:

            # pick the action that yields the greatest reward
            next_action_id = np.argmax(rewards)

        # otherwise the agent keeps exploring
        else:
            
            # if there's any unvisited state
            if len(unvisited_states) > 0:
                next_action_id = np.random.choice(unvisited_states)
            
            # otherwise pick one among all possible actions
            else:
                next_action_id = np.random.choice(np.arange(0,self.nb_actions))

        return next_action_id

    
    def kick_ball(self):
        """
        Trigger kicking motion.
        Inputs:
        -...
        Outputs:
        - ...
        """

        pass

    def update_policy(self, reward):
        """
        Handles ...
        Inputs:
        -...
        Outputs:
        - reward: reward previously obtained
        """

        pass



class Environment:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll, resolution):
        
        self.yaw_dof = JointDegreeOfFreedom(resolution, up_bound_yaw, low_bound_yaw)
        self.roll_dof = JointDegreeOfFreedom(resolution, up_bound_roll, low_bound_roll)

        # initialize environment
        # define default reward
        self.default_reward = -20

        # environment
        self.environment = np.ones((resolution, resolution)) * self.default_reward

        # Map actions to directions to move on the grid: -1 -> reduce, +1 -> augment
        self.action_to_direction = {
            0: [1, 0],
            1: [0, 1],
            2: [-1, 0],
            3: [0, -1]
        }

    def move_to_next_state(self, current_state_id, action_id):
        """
        Given the ids of the current state and the action, compute the next state and indicate if it's within the boundaries of the environment
        Inputs:
        - current_state_id: linear index of the current state
        - action_id: linear index of the action
        Outputs:
        - next_state_id: linear index of the next state
        - is_in_bounds: boolean
        """

        # Compute the coordinates of the current state on the grid environment (linear index to 2D coordinates)
        current_state_coords = np.unravel_index(current_state_id, (self.resolution, self.resolution))

        # Get the direction of the movement based on the action id
        movement_vector = np.asarray(self.action_to_direction(action_id))
        
        # Compute the coordinates of the next state on the grid environment
        next_state_coords = current_state_coords + movement_vector

        # Calculate the linear index of the next state (2D coordinates to linear index)
        next_state_id = np.ravel_multi_index((next_state_coords[0], next_state_coords[1])) 

        # Check if the next state lies within the boundaries of the environment
        if (0 < next_state_coords[0] < self.resolution) and (0 < next_state_coords[1] < self.resolution):
            is_in_bounds = True
        else:
            is_in_bounds = False

        return next_state_id, is_in_bounds



    def log_reward(self,data):
        """
        Handles ...
        Inputs:
        -...
        Outputs:
        - ...
        """

        pass


class JointDegreeOfFreedom:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, resolution, upper_bound, lower_bound):

        self.lower_bound = 1
        self.upper_bound = upper_bound
        self.resolution = resolution
        self.current_state_id = None

    def step(self, action_id, current_state_id):
        """
        Handles ...
        Inputs:
        -...
        Outputs:
        - ...
        """

        pass