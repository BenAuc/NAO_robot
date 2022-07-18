
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
import traceback
import pickle


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
    Class implementing the agent that learns a policy as it navigates an environment and collect rewards
    @Inputs:
    -resolution of state variables
    -range of state variables
    -position of goal posts in field of view
    @Outputs:
    -instantiate an object Environment
    -instantiate an object Policy
    """

    def __init__(self, hip_joint_resolution, hip_joint_start_position, goalkeeper_resolution, goalkeeper_x, goal_lims, r_hip_roll_limits, r_knee_pitch_limits):

        # SOME ACLARATIONS:
        # - the state space is composed by two features: hip_joint_position and goalkeeper_position
        # - feature 1 (hip_joint_position) is quantized with hip_joint_resolution between hip_roll_lims[0] and hip_roll_lims[1]
        # - feature 2 (goalkeeper_x) is quantized with goalkeeper_resolution between the left post of the goal and the right post of the goal
        # - the knee_pitch_lims are only used later for robot movement
        # - the action space is composed by three actions: move-in, move-out, and kick

        # Initialize the actions dictionary the policy can use
        self.action_dictionary = {
            0: 'move-in',
            1: 'move-out',
            2: 'kick'
            }

        # define camera resolution
        self.cam_y_max = 240 - 1 # camera resolution
        self.cam_x_max = 320 - 1 # camera resolution

        # Declare the two features in the state-space: the leg displacement and the goalkeeper x position
        self.feature1 = np.linspace(r_hip_roll_limits[0], r_hip_roll_limits[1], hip_joint_resolution) # State feature 1: position of the hip joint = leg displacement
        self.feature2 = np.linspace(goal_lims[0], goal_lims[1], goalkeeper_resolution) # State feature 2: position of the goalkeeper x coordinate

        # Initialize the actions that the robot can perform
        self.HipRollDiscretized = np.linspace(r_hip_roll_limits[0], r_hip_roll_limits[1], hip_joint_resolution) # used for actions 'move-in' and 'move-out', thus resolution_leg
        self.KneePitchDiscretized = np.linspace(r_knee_pitch_limits[0], r_knee_pitch_limits[1], 2) # used for action 'kick', thus resolution = 2 (knee goes from back to front)
        
        # variable containing the environment the policy is about
        self.environment = Environment(hip_joint_resolution, goalkeeper_resolution, self.action_dictionary)
        
        # variable containing the policy the agent has so far learned
        self.policy = Policy(environment=self.environment)

        # Store the hip joint position and goalkeeper x position
        self.hip_joint_position = hip_joint_start_position
        self.goalkeeper_x = goalkeeper_x

        # compute the linear id of the agent's current state in the policy table 
        # based on the hip joint position given as start position vector and the position of goalkeeper
        current_state_coords = self.quantize_state(self.hip_joint_position, self.goalkeeper_x)
        self.current_state_id = self.environment.state_coords_to_state_id(current_state_coords)

        # Flag to differentiate between train and forward mode
        self.train_mode = True

        # Keep track of the reward
        self.reward_total = 0
        self.reward = []
        self.a1 = current_state_coords[0]
        self.a2 = 0
        # create topic publishers
        # self.jointStiffnessPub = rospy.Publisher("joint_stiffness", JointState, queue_size=1)
        # self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10) # Allow joint control
        
        # Start main loop
        self.main_loop()


    def main_loop(self):
        """
        Run the step() function until the 'kick' action is selected
        """

        while True:
            # Step the policy
            print("------------")
            print("New step")
            action_id = self.step()
            print("Action id: " + str(action_id))

            # Check if the agent has selected the 'kick' action
            if action_id == 2:
                break


    def quantize_state(self, hip_joint_position, goalkeeper_x):
        """
        Quantize the state hip_joint_position and the goalkeeper_x to find the coordinates of the state in the grid environment
        Inputs:
        - hip_joint_position: float containing the position of the hip joint
        - goalkeeper_x: float containing the x-coordinate of the goalkeeper
        Outputs:
        - state_coords: list containing the coordinates of the state on the grid environment
        """

        # quantize the state hip_joint_position and the goalkeeper_x
        state_coords = [0, 0]
        state_coords[0] = np.argmin(np.abs(self.feature1 - hip_joint_position))
        state_coords[1] = np.argmin(np.abs(self.feature2 - goalkeeper_x))

        return state_coords

        
##########################


    # def step(self):
    #     """
    #     Perform step of the agent in the environment: choose an action, execute the action on the robot, 
    #     measure the new state, collect the reward, and update the policy (if we are in train mode)
    #     Outputs:
    #     - action_id: Return the selected action id
    #     """

    #     ## Choose an action
    #     action_id = self.policy.select_next_action(self.current_state_id) # request computation of the id of the action to take given the current state

    #     ## Execute the action on the robot
    #     self.execute_action(action_id)

    #     ## Measure the new state
    #     self.hip_joint_position += 0.2 # TODO: get the hip joint position from the robot (using the joint_angles topic)
    #     next_state_coords = self.quantize_state(self.hip_joint_position, self.goalkeeper_x)
    #     next_state_id = self.environment.state_coords_to_state_id(next_state_coords)

    #     ## If in train mode, collect the reward and update the policy model
    #     if self.train_mode:
    #         # collect the reward
    #         reward = self.get_reward(action_id)
    #         self.reward.append(reward)
    #         self.reward_total += reward
    #         # update the policy
    #         self.policy.update_model(self.current_state_id, next_state_id, action_id, reward)

    #     ## Update the current state
    #     self.current_state_id = next_state_id

    #     return action_id

    # def execute_action(self, action_id):
    #     """
    #     Execute the action on the robot
    #     Inputs:
    #     - action_id: int containing the id of the action to execute
    #     """
            
    #     # Get the direction of the action
    #     action_direction = self.environment.action_id_to_direction(action_id) # [hip_movement, knee_movement]
    #     self.a1 += action_direction[0]
    #     self.a2 += action_direction[1]

    #     print('a1: ', self.a1)
    #     print('a2: ', self.a2)

    #     #Get action for NAO execution
    #     hip_roll = self.HipRollDiscretized[self.a1]
    #     knee_pitch = self.HipRollDiscretized[self.a2]

    #     self.set_joint_angles(hip_roll, "RHipRoll")
    #     rospy.sleep(0.2)
    #     self.set_joint_angles(knee_pitch, "RKneePitch")
    #     rospy.sleep(0.2)


##########################


    def step(self, current_state_variables):
        """
        Perform step of the agent in the environment: choose an action, execute the action on the robot, 
        measure the new state, collect the reward, and update the policy (if we are in train mode)
        Inputs:
        - current_state_variables: current state of state variable
        Outputs:
        - action_id: Return the selected action id
        """

        ## Choose an action
        action_id = self.policy.select_next_action(self.current_state_id) # request computation of the id of the action to take given the current state

        # Get the direction of the action
        action_direction = self.environment.action_id_to_direction(action_id) # [hip_movement, knee_movement]
        self.a1 = current_state_variables + action_direction[0]
        self.a2 += action_direction[1]

        print('a1: ', self.a1)
        print('a2: ', self.a2)

        #Get action for NAO execution
        hip_roll = self.HipRollDiscretized[self.a1]
        knee_pitch = self.HipRollDiscretized[self.a2]

        ## Measure the new state
        self.hip_joint_position = hip_roll # TODO: get the hip joint position from the robot (using the joint_angles topic)
        next_state_coords = self.quantize_state(self.hip_joint_position, self.goalkeeper_x)
        next_state_id = self.environment.state_coords_to_state_id(next_state_coords)

        ## temporary storage of state id to return it
        previous_state_id = self.current_state_id

        ## Update the current state
        self.current_state_id = next_state_id

        return hip_roll, action_id, previous_state_id


    def train(self, action_id, previous_state_id):
        """
        Collect reward and update the model of the policy
        Inputs:
        - action_id: int containing the id of the action that just got executed
        - previous_state_id: int containing the id of the state on the basis of which the action was selected
        """

        ## If in train mode, collect the reward and update the policy model

        # collect the reward
        reward = self.get_reward(action_id) # pause and wait for keyboard input
        self.reward.append(reward)
        self.reward_total += reward

        # update the policy
        self.policy.update_model(previous_state_id, self.current_state_id, action_id, reward)

        return 


    def get_reward(self, action_id):
        """
        Collect reward from a keyboard stroke from the experimenter
        Inputs:
        - action_id: int containing the id of the action that just got executed
        """
        
        # if action_id != 2:
        #     button = 0
        # else:
        #     button = input("Enter the key for the reward: ")

        button = input("Enter the key for the reward: ")
    
        """
        For each button push set the reward
        """
        if button == 0:
            reward = self.environment.default_R # default reward value
        elif button == 1:
            reward = self.environment.fall_R # reward for falling
        elif button == 2:
            reward = self.environment.score_R # reward for scoring a goal
        elif button == 3:
            reward = self.environment.block_R # reward for getting the ball blocked by the goalkeeper
        elif button == 5:
            reward = self.environment.miss_R  # reward for missing the goal

        return reward

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


    def load_policy(self,path):
        """
        Upload the policy learned during a prior training
        Inputs:
        -path: path to the .pickle file containing the policy
        Outputs:
        -self.policy: store the policy in the class variable
        """
        
        path_to_policy = '/home/bio/bioinspired_ws/src/tutorial_5/policy/environment.obj'
        self.policy= pickle.load(open(path_to_policy, 'rb'))


    def save_policy(self,policy):
        """
        Upload the policy learned during a prior training
        Inputs:
        policy to be stored
        """

        path_to_policy = '/home/bio/bioinspired_ws/src/tutorial_5/policy/environment.obj'
        pickle.dump(policy, open(path_to_policy, 'wb'))


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
            'visited_times': 0, # number of times the action has been taken in the current state
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
                    next_state, next_state_is_in_bounds = self.environment.move_to_next_state(state_id, action_id)
                    
                    # Update the properties of state-action transition to the next state
                    new_state[action_id]['next_state'] = next_state
                    new_state[action_id]['is_valid'] = next_state_is_in_bounds
                else:
                    new_state[action_id]['next_state'] = state_id # if the action is kick, the next state is the same as the current state
                    new_state[action_id]['is_valid'] = True

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
        value_list = np.zeros([self.num_rows, self.num_cols])
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
        ax.set_xlabel('X position of the goalkeeper')
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
        Select next action of the agent given current state.
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

        # Compute the difference in state coordinates
        delta_state_coords = np.array(next_state_coords) - np.array(current_state_coords)
        delta_feature1 = delta_state_coords[0]
        delta_feature2 = delta_state_coords[1]
        
        # Update the trees for the two state features and the reward
        self.environment.update_tree(1, action_id, current_state_coords, delta_feature1)  # feature 1
        self.environment.update_tree(2, action_id, current_state_coords, delta_feature2) # feature 2
        self.environment.update_tree(3, action_id, current_state_coords, reward) # reward

        # Update the transition probabilities and rewards for the two state features
        for state_id in range(self.nb_states):
            for action_id in range(self.nb_actions):
                
                # Translate the linear indices to actual coordinates on the grid
                state_coords = self.environment.state_id_to_coords(state_id) # [row, col]
                action_direction = self.environment.action_id_to_direction(action_id) # [yaw_dir, roll_dir]

                # Predict the probability of the next state given the current state and action with the decision trees
                pred_delta_feature1_prob = self.environment.predict_transition_probability(1, state_coords, action_id)
                pred_delta_feature2_prob = self.environment.predict_transition_probability(2, state_coords, action_id)
                
                # Update transition probability
                P = pred_delta_feature1_prob * pred_delta_feature2_prob
                self.explored_actions[state_id][action_id]['P'] = P

                # Predict the reward with the decision trees
                R = self.environment.reward_tree.predict(np.hstack((state_coords, action_id)).reshape(1, -1))
                self.explored_actions[state_id][action_id]['R'] = R
                
                # Update the Q-value
                sum_update = 0
                for state_id_Q in range(self.nb_states):
                    list_Qs = []
                    for action_id_Q in range(self.nb_actions):
                        list_Qs.append(self.explored_actions[state_id_Q][action_id_Q]['Q'])
                    sum_update += self.explored_actions[state_id_Q][action_id]['P'] * int(np.max(list_Qs))

                self.explored_actions[state_id][action_id]['Q'] = R + self.gamma * sum_update
                

class Environment:
    """
    Class implementing the environment navigated by the agent
    @Inputs:
    -resolution of the state variables
    -dictionary of possible actions
    @Outputs:
    -instantiate a Decision Tree Classifier
    """

    def __init__(self, resolution_leg, resolution_goalkeeper, action_dictionary):
        
        # Store the input arguments
        self.num_rows = resolution_leg  # number of rows in the grid
        self.num_cols = resolution_goalkeeper # number of columns in the grid
        self.num_states = self.num_rows * self.num_cols # number of states in the grid
        self.action_dictionary = action_dictionary # dictionary containing the actions
        self.num_actions = len(action_dictionary) # number of actions possible

        # Initialize the reward grid environment
        self.default_R = -1 # default reward value
        self.fall_R = -20   # reward for falling
        self.score_R = 20   # reward for scoring a goal
        self.block_R = -2   # reward for getting the ball blocked by the goalkeeper
        self.miss_R = -10   # reward for missing the goal (kicked the ball outside of the goal)

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
        self.delta_feature1 = np.zeros(1) # feature 1 change
        self.delta_feature2 = np.zeros(1) # feature 2 change
        self.delta_reward = np.zeros(1) # reward change

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


    def update_tree(self, tree_id, action_id, state_coords, new_delta):
        """
        Update the decision tree with ID tree_id with the action action_id and the current state current_state_id.
        Inputs:
        - tree_id: int containing the id of the tree to update
        - action_id: int containing the id of the action taken
        - state_coords: list containing the coordinates of the current state
        - new_delta: float containing the change in the state variable or reward
        """

        # Select the tree to update
        tree = self.tree_dict[tree_id]

        # Select the delta and extend it with the new one
        delta = self.delta_dict[tree_id]
        delta = np.append(delta, new_delta)
        self.delta_dict[tree_id] = delta

        # Add the state-action pair to the history (only the first time)
        if tree_id == 1:
            self.history = np.vstack((self.history, np.hstack((state_coords, action_id))))

        # Update the tree
        tree.fit(self.history, delta)
        self.tree_dict[tree_id] = tree


    def predict_transition_probability(self, tree_id, state_coords, action_id):
        """
        Predict the probability of transition from the current state to the next state.
        Inputs:
        - tree_id: int containing the id of the tree to update
        - state_coords: list containing the coordinates of the current state
        - action_id: int containing the id of the action taken
        Outputs:
        - probability: float containing the probability of transition
        """

        # Select the tree to predict
        tree = self.tree_dict[tree_id]

        # Predict the probability of transition
        probability = tree.predict_proba(np.hstack((state_coords, action_id)).reshape(1, -1))[0][0]

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
        except ValueError:
            # Happens when we have negative coordinates (out of bounds), so we return the current state
            next_state_id = current_state_id
            is_in_bounds = False

        # Check if the next state lies within the boundaries of the environment
        is_in_bounds = (0 <= next_state_coords[0] < self.num_rows) and (0 <= next_state_coords[1] < self.num_cols)

        return next_state_id, is_in_bounds
    
    
# BELOW CODE IS ONLY FOR TEST PURPOSES, TO BE ABLE TO RUN THE FILE FROM HOME
# WITHOUT NEEDING TO EXECUTE ROS-RELATED FILES
if __name__=='__main__':

    # instantiate class and start loop function
    try:

        print('Starting...')

        # Initialize the environment resolution for the 2 state features
        HIP_JOINT_RESOLUTION, GOALKEEPER_RESOLUTION = 5, 5 # Resolution for the quantization of the leg displacement and goalkeeper x coordinate
        r_hip_roll_limits = [-1, 1]
        r_knee_pitch_limits = [-1, 1]
        # TODO: Here we should actually pass the measured values of the robot and the blob
        hip_joint_start_position = 0.0 # Displacement of the leg (= hip roll) at the moment when we start the RL algorithm)
        goalkeeper_x = 1.0 # x coordinate of the goalkeeper (in the camera frame) -> measured as center of the red blob
        goal_lims = [0.0, 3.0] # x coordinates of the goal posts (in the camera frame) -> measured with the AruCo markers

        # Instantiate the agent, which also instantiates the policy and the environment
        agent = Agent(hip_joint_resolution=HIP_JOINT_RESOLUTION, 
            hip_joint_start_position=hip_joint_start_position,
            goalkeeper_resolution=GOALKEEPER_RESOLUTION, 
            goalkeeper_x=goalkeeper_x,
            goal_lims=goal_lims,
            r_hip_roll_limits = r_hip_roll_limits,
            r_knee_pitch_limits= r_knee_pitch_limits
            )
        # agent.policy.print_explored_actions()
        # agent.policy.plot()
        
        print('Done!')
        
    except Exception:
        traceback.print_exc()