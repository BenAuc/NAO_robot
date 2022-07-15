# TODOs:

## agent_environment_model.py
 - Connect the parameters that are used to initialize the Agent model with the measured values: 
   ![image](https://user-images.githubusercontent.com/95912004/179242012-02b5853a-09b2-40d8-b005-08867183844c.png)
   - `hip_joint_start_position`: hip roll at the moment when we start the RL algorithm
   - `goalkeeper_x`: x coordinate of the goalkeeper (in the camera frame) -> measured as center of the red blob
   - `goal_lims`: x coordinates of the goal posts (in the camera frame) -> measured with the AruCo markers
 - Find empirical limits of the hip roll (for finding the leg displacement) and of the knee (for kicking the ball):
   ![image](https://user-images.githubusercontent.com/95912004/179242566-0e656b6b-f539-45b6-b618-8b5994ef113b.png)
 - Update the `Agent.step()` function to work correctly
 - Create function for upright posture: we could use posture proxies for that (http://doc.aldebaran.com/2-1/naoqi/motion/alrobotposture.html) or simply `Agent.set_joint_angles()`
 - Create `training` and `forward` modes to train the RL algorithm or use it in forward modus
   - We will need a `give_reward` function for the user to tell the robot the reward for its action
 - Train the robot (needs to be done at the lab)
