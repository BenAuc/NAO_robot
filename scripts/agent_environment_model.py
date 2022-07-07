#!/usr/bin/env python

#################################################################
# file name: agent_environment_model.py
# author's name: Diego, Priya, Vildana, Benoit Auclair
# created on: 30-06-2022
# last edit: 07-07-2022 
# function: define the Agent, Policy, and Environment class
#################################################################


"""
Commented out by Diego to be able to develop from home

from hmac import new
import rospy
from geometry_msgs.msg import Point, PolygonStamped
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed, HeadTouch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv2.aruco as aruco
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, resolution, start_position, low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll):

        # variable containing the environment the policy is about
        # in this case this is the 2 degrees of freedom of the hip
        self.environment = Environment(low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll, resolution)
        
        # variable containing the policy the agent has so far learned
        self.policy = Policy(nb_actions=4, resolution=resolution, environment=self.environment)
        
        # compute the linear id of the agent's current state in the policy table based on the 2 hip joint states given as start position vector
        self.current_state_id = np.ravel_multi_index((start_position[0], start_position[1]), (resolution, resolution)) 

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
        # self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)
        # self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10) # Allow joint control


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

        # Set action to visited
        self.policy.explored_actions[self.current_state_id][action_id]['visited'] = True

        # Get the next state given the action
        self.current_state_id = self.policy.explored_actions[self.current_state_id][action_id]['next_state']

        """
        I think this is equivalent to what we did in the line above

        # Given the ids of the current state and the action, update the current state to the next state and check if it lies within environment boundaries
        self.current_state_id, is_in_bounds = self.policy.environment.move_to_next_state(self.current_state_id, action_id)

        # Raise an error if the computed state is not within the boundaries of the environment
        if not is_in_bounds:
            raise ValueError('The computed next state of the agent lies outside the boundaries of the environment.')
        """

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


class Policy:
    """
    Class implementing ...
    @Inputs:
    -...
    @Outputs:
    -...
    """

    def __init__(self, nb_actions, resolution, environment):
        
        # Initialize the parameters of the policy
        self.nb_actions = nb_actions        # number of actions
        self.resolution = resolution        # resolution of the environment
        self.nb_states = resolution ** 2    # number of states available in the environment
        
        # Create a table to store the values of the state-value function
        self.V = np.zeros((self.nb_states, 1)) # V(s) = 0 for all states

        # Create a policy table
        self.policy = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions # initialize the policy with uniform probability of taking each action

        # define e-greedy policy epsilon parameter (exploitation threshold)
        self.epsilon = 0.3 # threshold when the agent starts exploiting the knowledge of the environment rather than exploring
        
        self.environment = environment # environment object

        self.explored_actions = self.explore_environment() # list of size (nb_states, nb_actions) containing a state_action_pair for each state and action

    def new_state_action_pair(self):
        """
        Needs to be defined here and not as a class property, because then we would store a handle to the same dictionary instance. This way, we can obtain a new instance of the dictionary for each state-action pair.
        """

        # define state action pairs as dictionaries
        state_action_pair = {
            'reward': 0,        # reward obtained by the agent in the next state (after taking the action)
            'next_state': 0,    # linear id of next state after taking the action
            'visited': False,   # whether the action has already been taken in the current state
            'is_valid': True    # whether the next state is valid (i.e. it is within the boundaries of the environment)
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

                # Check if the action is valid (action doesn't lead the agent outside the boundaries of the environment)
                next_state, next_state_is_in_bounds, next_reward = self.environment.move_to_next_state(state_id, action_id)
                
                # Update the properties of state-action transition to the next state
                new_state[action_id]['next_state'] = next_state
                new_state[action_id]['is_valid'] = next_state_is_in_bounds
                new_state[action_id]['reward'] = next_reward

            # Append row to the list
            explored_actions.append(new_state)

        return explored_actions
        
    def print_explored_actions(self):
        """
        Print the explored_actions array
        """

        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                reward = self.explored_actions[s][a]['reward']
                next_state = self.explored_actions[s][a]['next_state']
                visited = self.explored_actions[s][a]['visited']
                is_valid = self.explored_actions[s][a]['is_valid']
                print('State: {}, Action: {}, Next State: {}, Next State Reward: {}, Visited: {}, Valid Transition: {}'.format(s, a, next_state, reward, visited, is_valid))

    def plot(self):
        """
        Plot the policy
        """

        draw_vals = True
        plt.rcParams['figure.dpi'] = 175
        plt.rcParams.update({'axes.edgecolor': (0.32,0.36,0.38)})
        plt.rcParams.update({'font.size': 4})
        plt.figure(figsize=(self.resolution, self.resolution))
        plt.imshow(1 - self.V.reshape(self.resolution, self.resolution), cmap='gray', interpolation='none', clim=(0,1))
        ax = plt.gca()
        ax.set_xticks(np.arange(self.resolution)-.5)
        ax.set_yticks(np.arange(self.resolution)-.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for state_id in range(len(self.policy)):
            x = state_id%self.resolution
            y = int(state_id/self.resolution)
            current_policy = self.policy[state_id] # get the policy for the current state
            gray = np.array((0.32,0.36,0.38))
            # Draw arrows for the actions
            if self.explored_actions[state_id][0]['is_valid']: plt.arrow(x, y, float(current_policy[0])*.84, 0.0,  color=gray+0.2*(1-self.V[state_id]), head_width=0.1, head_length=0.1) # right
            if self.explored_actions[state_id][1]['is_valid']: plt.arrow(x, y, 0.0, float(current_policy[1])*-.84, color=gray+0.2*(1-self.V[state_id]), head_width=0.1, head_length=0.1) # up
            if self.explored_actions[state_id][2]['is_valid']: plt.arrow(x, y, float(current_policy[2])*-.84, 0.0, color=gray+0.2*(1-self.V[state_id]), head_width=0.1, head_length=0.1) # left
            if self.explored_actions[state_id][3]['is_valid']: plt.arrow(x, y, 0.0, float(current_policy[3])*.84,  color=gray+0.2*(1-self.V[state_id]), head_width=0.1, head_length=0.1) # down

            if draw_vals and self.V[state_id]>0:
                vstr = '{0:.1e}'.format(self.V[state_id]) if self.resolution == 8 else '{0:.6f}'.format(self.V[state_id])
                plt.text(x-0.45,y+0.45, vstr, color=(gray*self.V[state_id]), fontname='OpenSans')
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

        # intialize a list of rewards of the state-action pairs -> needed for greedy policy
        rewards = []

        # check whether each of the actions is valid and unvisited
        for action in range(self.nb_actions):

            # check if the action is valid
            if self.explored_actions[state][action]['is_valid']:

                # add the reward of the state-action pair to the list of rewards
                rewards.append(self.explored_actions[state][action]['reward'])
                    
                # check if the action has already been visited
                if (not self.explored_actions[state][action]['visited']):

                    # add the action to the list of unvisited actions
                    unvisited_actions.append(action)
            else:
                # if the action is not valid, add a reward of -1 to the list of rewards
                rewards.append(-1)

        # generate random number between 0 and 1 (decides whether to use the greedy or random policy)
        greedy_probability = np.random.randn(1)

        # if larger than e-greedy threshold then go with greedy policy
        if greedy_probability > self.epsilon:

            # pick the action that yields the greatest reward
            next_action_id = np.argmax(rewards)

        # otherwise pick a random action
        else:
            
            # if there's any unvisited action, pick one of them randomly
            if len(unvisited_actions) > 0:
                # pick a random action from the list of unvisited actions
                next_action_id = np.random.choice(unvisited_actions)
            
            # otherwise pick one among all valid actions
            else:
                # pick a random action from the list of valid actions, where rewards is not -1
                next_action_id = np.random.choice(np.arange(0, self.nb_actions)[np.where(rewards != -1)])

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
        self.resolution = resolution

        # initialize environment
        # define default reward
        self.default_reward = 0

        # environment
        self.reward_matrix = np.ones((resolution, resolution)) * self.default_reward

        # Map actions to directions to move on the grid: -1 -> reduce, +1 -> augment
        self.action_to_direction = {
            0: [0, 1],  # right
            1: [-1, 0], # up
            2: [0, -1], # left
            3: [1, 0]   # down
        }

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
        current_state_coords = np.unravel_index(current_state_id, (self.resolution, self.resolution))
        
        # Get the direction of the movement based on the action id
        movement_vector = np.asarray(self.action_to_direction[action_id])
        
        # Compute the coordinates of the next state on the grid environment
        next_state_coords = current_state_coords + movement_vector

        # Try to convert the coordinates of the next state to a linear index
        try:
            # Calculate the linear index of the next state (2D coordinates to linear index)
            next_state_id = np.ravel_multi_index(next_state_coords, (self.resolution, self.resolution))

            # Get reward at the next state
            reward = self.reward_matrix[next_state_coords[0], next_state_coords[1]]
        except ValueError:
            # Happens when we have negative coordinates (out of bounds), so we return the current state
            next_state_id = current_state_id
            is_in_bounds = False
            reward = -1

            return next_state_id, is_in_bounds, reward

        # Check if the next state lies within the boundaries of the environment
        is_in_bounds = (0 <= next_state_coords[0] < self.resolution) and (0 <= next_state_coords[1] < self.resolution)

        return next_state_id, is_in_bounds, reward



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
    
    
# BELOW CODE IS ONLY FOR TEST PURPOSES, TO BE ABLE TO RUN THE FILE FROM HOME
# WITHOUT NEEDING TO EXECUTE ROS-RELATED FILES
if __name__=='__main__':

    # instantiate class and start loop function
    try:

        # instantiate the agent
        start_position = [2, 2]
        low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll = -np.pi, np.pi, -np.pi, np.pi
        resolution = 5
        agent = Agent(resolution, start_position, low_bound_yaw, up_bound_yaw, low_bound_roll, up_bound_roll)
        agent.policy.print_explored_actions()
        agent.policy.plot()
        
    except:
        pass