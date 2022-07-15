#!/usr/bin/env python

#################################################################
# file name: agent_environment_model.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 30-06-2022
# last edit: 14-07-2022 (Benoit): added joint limits to ROS parameter server
# function: define the Agent, Policy, and Environment class
#################################################################



# Commented out by Diego to be able to develop from home

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
from sklearn import tree


"""
Dictionary of RL terminology:
- reward: reward obtained by the agent in a state
- return: total future reward with discounting (gamma)
- state: current position of the agent on the grid
- action: movement made by the agent in a state
- next_state: next position of the agent after an action
- state-value function: expected return at a state when following a given policy -> V(s)
- action-value function: expected state-value when choosing an action with a given policy -> Q(s,a)
- policy: probability of taking each action in a given state -> p(a|s)
"""


class Agent:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, hip_joint_resolution, hip_joint_start_position, goal_keeper_resolution):

        # SOME ACLARATIONS:
        # - the state space is composed by two features: hip_joint_position and goal_keeper_position
        # - feature 1 (hip_joint_position) is quantized with hip_joint_resolution between hip_roll_lims[0] and hip_roll_lims[1]
        # - feature 2 (ball_x) is quantized with goal_keeper_resolution between 0 and self.cam_x_max
        # - the knee_pitch_lims are only used later for robot movement
        # - the action space is composed by three actions: move-in, move-out, and kick

        # Initialize the actions dictionary the policy can use
        self.action_dictionary = {
            0: 'move-in',
            1: 'move-out',
            2: 'kick'
            }

        # define joint limits
        self.r_hip_pitch_limits = rospy.get_param("joint_limits/right_hip/pitch")
        self.r_hip_roll_limits = rospy.get_param("joint_limits/right_hip/roll")

        self.r_ankle_pitch_limits = rospy.get_param("joint_limits/right_ankle/pitch")
        self.r_knee_pitch_limits = rospy.get_param("joint_limits/right_knee/pitch")

        self.joint_limit_safety_factor = rospy.get_param("joint_limits/safety")[0]

        # define camera resolution
        self.cam_y_max = 240 - 1 # camera resolution
        self.cam_x_max = 320 - 1 # camera resolution

        # Declare the two features in the state-space: the leg displacement and the ball x position
        self.feature1 = np.linspace(self.r_hip_roll_limits[0], self.r_hip_roll_limits[1], hip_joint_resolution)
        self.feature2 = np.linspace(0, self.cam_x_max, goal_keeper_resolution)

        # Initialize the actions that the robot can perform
        self.HipRollDiscretized = np.linspace(self.r_hip_roll_limits[0], self.r_hip_roll_limits[1], hip_joint_resolution) # used for actions 'move-in' and 'move-out', thus resolution_leg
        self.KneePitchDiscretized = np.linspace(self.r_knee_pitch_limits[0], self.r_knee_pitch_limits[1], 2)        # used for action 'kick', thus resolution = 2 (knee goes from back to front)

        # variable containing the environment the policy is about
        self.environment = Environment(hip_joint_resolution, goal_keeper_resolution, self.action_dictionary)
        
        # variable containing the policy the agent has so far learned
        self.policy = Policy(environment=self.environment)

        # compute the linear id of the agent's current state in the policy table 
        # based on the hip joint position given as start position vector
        # and the position of goal keeper assumed to be in the middle of the field of view
        current_state_coords = self.quantize_state(hip_joint_start_position, int(self.cam_x_max / 2))
        self.current_state_id = self.environment.state_coords_to_state_id(current_state_coords)

        # flag to indicate that the agent is in position and ready to kick a goal
        # TODO: we can couple this flat to the press of a button to allow us to trigger the node when we want it to execute actions
        self.readiness = False

        # create topic publishers
        # self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)
        # self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10) # Allow joint control

    def quantize_state(self, hip_joint_position, ball_x):
        """
        Quantize the state hip_joint_position and the ball_x to find the coordinates of the state in the grid environment
        Inputs:
        - hip_joint_position: float containing the position of the hip joint
        - ball_x: float containing the x-coordinate of the ball
        Outputs:
        - state_coords: list containing the coordinates of the state on the grid environment
        """

        # quantize the state hip_joint_position and the ball_x
        state_coords = [0, 0]
        state_coords[0] = np.argmin(np.abs(self.feature1 - hip_joint_position))
        state_coords[1] = np.argmin(np.abs(self.feature2 - ball_x))

        return state_coords

    def step(self, goal_keeper_position, goal_position):
        """
        Perform step.
        Inputs:
        -goal_keeper_position: numpy array 1x2 of pixel coordinates (x,y) of goal keeper in camera field of view
        -goal_position: numpy array 2x2 of pixel coordinates (x,y) of edges of goal in camera field of view
        Outputs:
        -computes next state of the agent
        """
        # TODO: the arguments goal_keeper_position, goal_position still need to be integrated into the computations
        # we basically don't control the state of the position of goal keeper and goal
        # so we would need to recompute the current_state each time based on input from NAO's camera & Aruco markers

        # request computation of the id of the action to take given the current state
        action_id = self.policy.select_next_action(self.current_state_id)

        # Set action to visited
        self.policy.explored_actions[self.current_state_id][action_id]['visited'] = True

        # Get the next state given the action
        self.current_state_id = self.policy.explored_actions[self.current_state_id][action_id]['next_state']

        return

    def set_joint_angles(self, head_angle, joint_name):
        """
        Handles incoming motor command to set the joint state.
        Inputs:
        -joint_name: string containing the joint name
        -head_angle: float containing the desired state
        Outputs:
        -triggers the motion
        """

        """
        Commented out by Diego to be able to develop from home

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)
        """
        joint_angles_to_set = 0 # DELETE THIS LINE

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


    def set_upright_position(self):
        """
        Sets NAO in upright position at the start
        Inputs:
        -TBD
        Outputs:
        -TBD
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

    def __init__(self, environment):
        
        # Initialize the parameters of the policy
        self.environment = environment # environment object 
        self.action_dictionary = self.environment.action_dictionary # dictionary containing the actions the policy can perform
        self.nb_actions = len(self.action_dictionary)        # number of actions
        self.num_rows = self.environment.num_rows # number of rows in the grid environment
        self.num_cols = self.environment.num_cols # number of columns in the grid environment
        self.nb_states = self.num_rows * self.num_cols    # number of states available in the environment
        
        # Create a policy table
        self.policy = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions # initialize the policy with uniform probability of taking each action
        
        # define e-greedy policy epsilon parameter (exploitation threshold)
        self.epsilon = 0.3 # threshold when the agent starts exploiting the knowledge of the environment rather than exploring
        
        self.explored_actions = self.explore_environment() # list of size (nb_states, nb_actions) containing a state_action_pair for each state and action
        
        # Initialize learning parameters
        self.alpha = 0.1 # learning rate alpha for the Q-learning algorithm
        self.gamma = 0.99 # discount factor gamma for the Q-learning algorithm


    def new_state_action_pair(self):
        """
        Needs to be defined here and not as a class property, because then we would store a handle to the same dictionary instance. This way, we can obtain a new instance of the dictionary for each state-action pair.
        """

        # define state action pairs as dictionaries
        state_action_pair = {
            'next_state': 0,    # linear id of next state after taking the action
            'visited': False,   # whether the action has already been taken in the current state
            'is_valid': True,   # whether the next state is valid (i.e. it is within the boundaries of the environment)
            'R': 0,             # reward R obtained by the agent in the next state (after taking the action)
            'P': 0,             # probability P of the action being taken in the next state (trainsition model: P(s'|s,a) = P(s'|s)P(a|s))
            'Q': 0              # Q-value Q(s,a)
            }

        return state_action_pair


    def explore_environment(self):
        """
        Visit all possible state-action pairs and fill the explored_actions array
        Outputs:
        - explored_actions: list of size (nb_states, nb_actions) containing a state_action_pair for each state and action
        """

        explored_actions = [] # array of size (nb_states, nb_actions)

        # Fill the self.explored_actions array of size (nb_states, nb_actions) with self.state_action_pair's at each entry
        for state_id in range(self.nb_states):
            # Create new row
            new_state = []
            # Loop through columns
            
            for action_id in range(self.nb_actions):
                # Store self.state_action_pair in the cell
                new_state.append(self.new_state_action_pair())

                # Check if the action is a kick action (action_id == 2), if so, this action is not used to explore the enviornment
                if action_id != 2:
                    # Check if the action is valid (action doesn't lead the agent outside the boundaries of the environment)
                    next_state, next_state_is_in_bounds, next_reward = self.environment.move_to_next_state(state_id, action_id)
                    
                    # Update the properties of state-action transition to the next state
                    new_state[action_id]['next_state'] = next_state
                    new_state[action_id]['is_valid'] = next_state_is_in_bounds
                    # new_state[action_id]['R'] = next_reward
                else:
                    new_state[action_id]['next_state'] = state_id # if the action is kick, the next state is the same as the current state
                    new_state[action_id]['is_valid'] = False

            # Append row to the list
            explored_actions.append(new_state)

        return explored_actions
        
    def print_explored_actions(self):
        """
        Print the explored_actions array
        """

        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                x, y = self.environment.state_id_to_coords(s)
                a_name = self.action_dictionary[a]
                reward = self.explored_actions[s][a]['R']
                next_state = self.explored_actions[s][a]['next_state']
                visited = self.explored_actions[s][a]['visited']
                is_valid = self.explored_actions[s][a]['is_valid']
                print('State: (ID: {}, 0-based coordinates: [{}, {}]), Action: {}, Next State: {}, Next State Reward: {}, Visited: {}, Valid Transition: {}\n'.format(s, x, y, a_name, next_state, reward, visited, is_valid))

    def plot(self):
        """
        Plot the policy
        """

        # Whatever values you want to plot
        value_list = self.environment.reward_grid.flatten()

        draw_vals = True
        plt.rcParams['figure.dpi'] = 175
        plt.rcParams.update({'axes.edgecolor': (0.32,0.36,0.38)})
        plt.rcParams.update({'font.size': 4})
        plt.figure(figsize=(self.num_rows, self.num_cols))
        plt.imshow(1 - value_list.reshape(self.num_rows, self.num_cols), cmap='gray', interpolation='none', clim=(0,1))
        ax = plt.gca()
        ax.set_xticks(np.arange(self.num_rows)-.5)
        ax.set_yticks(np.arange(self.num_cols)-.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('X position of the ball')
        ax.set_ylabel('Displacement of the leg')
        for state_id in range(len(self.policy)):
            y, x = self.environment.state_id_to_coords(state_id)
            current_policy = self.policy[state_id] # get the policy for the current state
            gray = np.array((0.32,0.36,0.38))
            # Draw arrows for the actions
            if self.explored_actions[state_id][0]['is_valid']: plt.arrow(x, y, 0.0, float(current_policy[0])*.84,  color=gray+0.2*(1-value_list[state_id]), head_width=0.1, head_length=0.1) # move-in (shown as arrown down)
            if self.explored_actions[state_id][1]['is_valid']: plt.arrow(x, y, 0.0, float(current_policy[1])*-.84, color=gray+0.2*(1-value_list[state_id]), head_width=0.1, head_length=0.1) # move-out (shown as arrow up)
            if self.explored_actions[state_id][2]['is_valid']: plt.arrow(x, y, float(current_policy[2])*.84, 0.0, color=gray+0.2*(1-value_list[state_id]), head_width=0.1, head_length=0.1) # kick (shown as arrow right)

            if draw_vals and value_list[state_id]>0:
                vstr = '{0:.1e}'.format(value_list[state_id]) if self.num_rows == 8 else '{0:.6f}'.format(value_list[state_id])
                plt.text(x-0.45,y+0.45, vstr, color=(gray*value_list[state_id]))
        plt.grid(color=(0.42,0.46,0.48), linestyle=':')
        ax.set_axisbelow(True)
        ax.tick_params(color=(0.42,0.46,0.48),which='both',top=False,left=False,right=False,bottom=False)
        plt.show()


    def select_next_action(self, state):
        """
        Handles ...
        Inputs:
        - state: int containing the id of the current state
        Outputs:
        - next_action_id: int containing the id of the action to take
        """

        # intialize a list of unvisited actions for the current state -> needed for random selection of an action
        unvisited_actions = []

        # intialize a list of Q-values of the state-action pairs -> needed for greedy policy
        Q_list = []

        # check whether each of the actions is valid and unvisited
        for action in range(self.nb_actions):

            # check if the action is valid (action doesn't lead the agent outside the boundaries of the environment)
            if self.explored_actions[state][action]['is_valid']:

                # add the Q-value of the state-action pair to the list Q_list
                Q_list.append(self.explored_actions[state][action]['Q'])
                    
                # check if the action has already been visited
                if (not self.explored_actions[state][action]['visited']):

                    # add the action to the list of unvisited actions
                    unvisited_actions.append(action)
            else:
                # if the action is not valid, add a Q of -1 to the list Q_list
                Q_list.append(-1)

        # generate random number between 0 and 1 (decides whether to use the greedy or random policy)
        greedy_probability = np.random.randn(1)

        # if larger than e-greedy threshold then go with greedy policy
        if greedy_probability > self.epsilon:

            # pick the action that yields the greatest Q
            next_action_id = np.argmax(Q_list)

        # otherwise pick a random action
        else:
            
            # if there's any unvisited action, pick one of them randomly
            if len(unvisited_actions) > 0:
                # pick a random action from the list of unvisited actions
                next_action_id = np.random.choice(unvisited_actions)
            
            # otherwise pick one among all valid actions
            else:
                # pick a random action from the list of valid actions, where Q_list is not -1
                next_action_id = np.random.choice(np.arange(0, self.nb_actions)[np.where(Q_list != -1)])

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


    def home_after_kick_ball(self):
        """
        Returns to home position after kicking the ball.
        Inputs:
        -...
        Outputs:
        - ...
        """

        pass


    def update_model(self, current_state_id, next_state_id, action_id, reward):
        """
        Works the same way as the function with the same name in the paper.
        Inputs:
        - current_state_id: int containing the id of the current state
        - next_state_id: int containing the id of the next state
        - action_id: int containing the id of the action taken
        - reward: float containing the reward received
        Outputs:
        - Q: numpy array containing the Q-values of the current state-action pairs
        """

        # Get the state coordinates and action direction vector
        current_state_coords = self.environment.state_id_to_coords(current_state_id) # [row, col]
        next_state_coords = self.environment.state_id_to_coords(next_state_id) # [row, col]
        action_direction = self.environment.action_id_to_direction(action_id) # [yaw_dir, roll_dir]

        # Compute the difference in state coordinates
        delta_state_coords = next_state_coords - current_state_coords
        delta_feature1 = delta_state_coords[0]
        delta_feature2 = delta_state_coords[1]
        
        # Update the trees for the two state features and the reward
        self.environment.update_tree(1, action_direction, current_state_coords, delta_feature1)  # feature 1
        self.environment.update_tree(2, action_direction, current_state_coords, delta_feature2) # feature 2
        self.environment.update_tree(3, action_direction, current_state_coords, reward) # reward

        # Update the transition probabilities and rewards for the two state features
        for state_id in range(self.nb_states):
            for action_id in range(self.nb_actions):
                # Translate the linear indices to actual coordinates on the grid
                state_coords = self.environment.state_id_to_coords(state_id) # [row, col]
                action_direction = self.environment.action_id_to_direction(action_id) # [yaw_dir, roll_dir]

                # Predict the probability of the next state given the current state and action with the decision trees
                pred_delta_feature1_prob = self.environment.predict_transition_probability(1, state_coords, action_direction)
                pred_delta_feature2_prob = self.environment.predict_transition_probability(2, state_coords, action_direction)

                # Update transition probability
                P = pred_delta_feature1_prob * pred_delta_feature2_prob
                self.explored_actions[state_id][action_id]['P'] = P

                # Predict the reward with the decision trees
                R = self.environment.reward_tree.predict(np.hstack((state_coords, action_direction)))
                self.explored_actions[state_id][action_id]['R'] = R

                # Update the Q-value
                self.explored_actions[state_id][action_id]['Q'] = R
                for state_id_Q in range(self.nb_states):
                    self.explored_actions[state_id][action_id]['Q'] += self.gamma * self.explored_actions[state_id_Q][action_id]['P'] * np.max(self.explored_actions[state_id_Q][:]['Q'])




class Environment:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, resolution_leg, resolution_ball, action_dictionary):
        
        # Store the input arguments
        self.num_rows = resolution_leg  # number of rows in the grid
        self.num_cols = resolution_ball # number of columns in the grid
        self.num_states = self.num_rows * self.num_cols # number of states in the grid
        self.action_dictionary = action_dictionary # dictionary containing the actions
        self.num_actions = len(action_dictionary) # number of actions possible

        # Initialize the reward grid environment
        self.default_R = -1 # default reward value
        self.fall_R = -20   # reward for falling
        self.score_R = 20   # reward for scoring a goal
        self.block_R = -2   # reward for getting the ball blocked by the goalkeeper
        self.miss_R = -10   # reward for missing the goal (kicked the ball outside of the goal)
        self.reward_grid = np.ones((self.num_rows, self.num_cols)) * self.default_R # reward grid

        # Map actions to directions to move on the grid: -1 -> reduce, +1 -> augment
        self.action_to_direction = {
            # First column: change in hip roll, second column: change in knee pitch
            0: [1, 0],  # move-in
            1: [-1, 0], # move-out
            2: [0, 1], # kick
        }

        ## Initialize the decision trees
        self.feature1_tree = tree.DecisionTreeClassifier() # Predict probability of change in state variable 1
        self.feature2_tree = tree.DecisionTreeClassifier() # Predict probability of change in state variable 2
        self.reward_tree = tree.DecisionTreeClassifier() # Predict average reward

        # Initialize empty history of state-action pairs (all information is relative, not absolute)
        self.history = np.zeros(3) # [feature1, feature2, action_id]
        self.delta_feature1 = np.zeros(0) # feature 1 change
        self.delta_feature2 = np.zeros(0) # feature 2 change
        self.delta_reward = np.zeros(0) # reward change

        # Give an ID to each tree and to each delta
        self.tree_dict = {
            1: self.feature1_tree,
            2: self.feature2_tree,
            3: self.reward_tree
        }

        self.delta_dict = {
            1: self.delta_feature1,
            2: self.delta_feature2,
            3: self.delta_reward
        }



    def update_tree(self, tree_id, action_direction, state_coords, new_delta):
        """
        Update the decision tree with ID tree_id with the action action_id and the current state current_state_id.
        Inputs:
        - tree_id: int containing the id of the tree to update
        - action_direction: list containing the direction vector of the action
        - state_coords: list containing the coordinates of the current state
        - new_delta: float containing the change in the state variable or reward
        """

        # Select the tree to update
        tree = self.tree_dict[tree_id]

        # Select the delta and extend it with the new one
        delta = self.delta_dict[tree_id]
        delta = np.append(delta, new_delta)

        # Add the state-action pair to the history
        self.history = np.vstack((self.history, np.hstack((state_coords, action_direction))))

        # Update the tree
        tree.fit(self.history, delta)

    def predict_transition_probability(self, tree_id, state_coords, action_direction):
        """
        Predict the probability of transition from the current state to the next state.
        Inputs:
        - tree_id: int containing the id of the tree to update
        - state_coords: list containing the coordinates of the current state
        - action_direction: list containing the direction vector of the action
        Outputs:
        - probability: float containing the probability of transition
        """

        # Select the tree to predict
        tree = self.tree_dict[tree_id]

        # Predict the probability of transition
        probability = tree.predict_proba(np.hstack((state_coords, action_direction)))[0][0]

        return probability


    def state_id_to_coords(self, state_id):
        """
        Get the state coordinates corresponding to the state_id.
        Inputs:
        - state_id: int containing the id of the state
        Outputs:
        - state: numpy array containing the state
        """

        # Get the state corresponding to the state_id
        state_coords = np.unravel_index(state_id, (self.num_rows, self.num_cols))

        return state_coords

    def state_coords_to_state_id(self, state_coords):
        """
        Get the state_id corresponding to the state_coords.
        Inputs:
        - state_coords: list containing the coordinates of the state
        Outputs:
        - state_id: int containing the id of the state
        """

        # Get the state_id corresponding to the state_coords
        state_id = np.ravel_multi_index(state_coords, (self.num_rows, self.num_cols))

        return state_id


    def action_id_to_direction(self, action_id):
        """
        Get the action corresponding to the action_id.
        Inputs:
        - action_id: int containing the id of the action
        Outputs:
        - action: numpy array containing the action
        """
        
        # Get the action corresponding to the action_id
        action = np.asarray(self.action_to_direction[action_id])

        return action


    def move_to_next_state(self, current_state_id, action_id):
        """
        Given the ids of the current state and the action, compute the next state and indicate if it's within the boundaries of the environment
        Inputs:
        - current_state_id: linear index of the current state
        - action_id: linear index of the action
        Outputs:
        - next_state_id: linear index of the next state
        - is_in_bounds: boolean whether the next state is within the boundaries of the environment
        - reward: int containing the reward at the next state
        """
        
        # Compute the coordinates of the current state on the grid environment (linear index to 2D coordinates)
        current_state_coords = self.state_id_to_coords(current_state_id)
        
        # Get the direction of the movement based on the action id
        movement_vector = self.action_id_to_direction(action_id)
        
        # Compute the coordinates of the next state on the grid environment
        next_state_coords = current_state_coords + movement_vector
        
        # Try to convert the coordinates of the next state to a linear index
        try:
            # Calculate the linear index of the next state (2D coordinates to linear index)
            next_state_id = self.state_coords_to_state_id(next_state_coords)
        
            # Get reward at the next state
            reward = self.reward_grid[next_state_coords[0], next_state_coords[1]]
        except ValueError:
            # Happens when we have negative coordinates (out of bounds), so we return the current state
            next_state_id = current_state_id
            is_in_bounds = False
            reward = -1

            return next_state_id, is_in_bounds, reward

        # Check if the next state lies within the boundaries of the environment
        is_in_bounds = (0 <= next_state_coords[0] < self.num_rows) and (0 <= next_state_coords[1] < self.num_cols)

        return next_state_id, is_in_bounds, reward

'''
DEPRECATED: @Benoit, I don't think we need this anymore...

class JointDegreeOfFreedom:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, resolution, upper_bound, lower_bound):

        self.resolution = resolution
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.state_space = np.linspace(lower_bound, upper_bound, resolution)

    def step(self, action_id, current_state_id):
        """
        Handles ...
        Inputs:
        -...
        Outputs:
        - ...
        """

        pass

'''
    
    
# BELOW CODE IS ONLY FOR TEST PURPOSES, TO BE ABLE TO RUN THE FILE FROM HOME
# WITHOUT NEEDING TO EXECUTE ROS-RELATED FILES
if __name__=='__main__':

    # instantiate class and start loop function
    try:
        print('Starting...')

        # Initialize the environment variables
        RESOLUTION_LEG, RESOLUTION_BALL = 5, 5 # Resolution for the quantization of the leg displacement and ball x coordinate
        HIP_ROLL_LIMS = [-np.pi/2, np.pi/2] # Limits of the hip joint roll
        KNEE_PITCH_LIMS = [-np.pi/2, np.pi/2] # Limits of the knee joint pitch
        LEG_DISPLACEMENT_LIMS = [-np.pi/2, np.pi/2] # Limits of the leg displacement
        CAM_X_LIMS = [1, 200] # Limits of the camera x coordinate

        # TODO: Here we should actually pass the measured values of the robot and the blob
        leg_displacement = 1.0 # Displacement of the leg
        ball_x = 1.0 # x coordinate of the ball

        # Instantiate the agent, which also instantiates the policy and the environment
        agent = Agent(resolution_leg=RESOLUTION_LEG, 
            resolution_ball=RESOLUTION_BALL, 
            leg_displacement_lims=LEG_DISPLACEMENT_LIMS,
            cam_x_lims=CAM_X_LIMS,
            leg_displacement=leg_displacement, 
            ball_x=ball_x,
            hip_roll_lims=HIP_ROLL_LIMS, 
            knee_pitch_lims=KNEE_PITCH_LIMS
            )
        agent.policy.print_explored_actions()
        agent.policy.plot()
        
        print('Done!')
        
    except:
        pass