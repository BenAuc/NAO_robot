# TODOs:

## agent_environment_model.py
 - Connect the parameters that are used to initialize the Agent model with the measured values: 
   ![image](https://user-images.githubusercontent.com/95912004/179242012-02b5853a-09b2-40d8-b005-08867183844c.png)
   - `hip_joint_start_position`: hip roll at the moment when we start the RL algorithm
   - `goalkeeper_x`: x coordinate of the goalkeeper (in the camera frame) -> measured as center of the red blob
   - `goal_lims`: x coordinates of the goal posts (in the camera frame) -> measured with the AruCo markers
 - Find empirical limits of the hip roll (for finding the leg displacement) and of the knee (for kicking the ball):
   ![image](https://user-images.githubusercontent.com/95912004/179242566-0e656b6b-f539-45b6-b618-8b5994ef113b.png)
 - Create function for upright posture: we could use posture proxies for that (http://doc.aldebaran.com/2-1/naoqi/motion/alrobotposture.html) or simply `Agent.set_joint_angles()`
 - Write `Agent.execute_action()` to execute the movement on the robot, according to the selected action id
   ![image](https://user-images.githubusercontent.com/95912004/179284462-84d2415f-b95c-41af-8de0-e80e63fb5ca1.png)
 - Connect `self.hip_joint_position` with a subscriber to the corresponding topic
   ![image](https://user-images.githubusercontent.com/95912004/179284753-c56603d3-9f0e-4cc5-b987-3deb09f5f8de.png)
   On the image, we see a portion of the step function. After the call to `execute_action(action_id)` (see previous point), we expect the hip joint position to change.
   Thus, we fetch it from the rostopic and compute our new state. The first line on this image is hardcoded to "fake" a new measurement, since we don't know the real value yet.
 - Write the functions to plot the training progress over the episodes
 - **Train the robot (needs to be done at the lab)**
