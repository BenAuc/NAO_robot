#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed,Bumper,HeadTouch
from sensor_msgs.msg import Image,JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2

class Central:


    def __init__(self):
        # initialize class variables
        self.joint_names = []
        self.joint_angles = []
        self.joint_velocities = []
        self.jointPub = 0
        self.stiffness = False  
        self.head_touch = False
        self.r_arm_touch = False

        pass


    def key_cb(self,data):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def joints_cb(self,data):
        #rospy.loginfo("joint states "+str(data.name)+str(data.position))
        # store current joint information in class variables
        self.joint_names = data.name 
        self.joint_angles = data.position
        self.joint_velocities = data.velocity

        pass

    def bumper_cb(self,data):
        rospy.loginfo("bumper: "+str(data.bumper)+" state: "+str(data.state))
        if data.bumper == 0:
            self.stiffness = True
        elif data.bumper == 1:
            self.stiffness = False

    def touch_cb(self,data):
        rospy.loginfo("touch button: "+str(data.button)+" state: "+str(data.state))

        if data.button == 2 and data.state == 1 and not self.head_touch:
            self.head_touch = True
            print("*********")
            print("starting repetitive left arm motion routine")

        else:
            if data.button == 2 and data.state == 1 and self.head_touch:
                self.head_touch = False
                print("stopping repetitive left arm motion routine")
                print("*********")

        #if data.button == 3 and not self.r_arm_touch:
            #self.head_touch = True

    def image_cb(self,data):
        bridge_instance = CvBridge()
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
        
        cv2.imshow("image window",cv_image)
        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly

    # sets the stiffness for all joints. can be refined to only toggle single joints, set values between [0,1] etc
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

    def set_joint_angles(self,head_angle,joint_name):

        joint_angles_to_set = JointAnglesWithSpeed()
        joint_angles_to_set.joint_names.append(joint_name) # each joint has a specific name, look into the joint_state topic or google
        joint_angles_to_set.joint_angles.append(head_angle) # the joint values have to be in the same order as the names!!
        joint_angles_to_set.relative = False # if true you can increment positions
        joint_angles_to_set.speed = 0.1 # keep this low if you can
        self.jointPub.publish(joint_angles_to_set)
        
    def left_arm_home(self):

        self.set_stiffness(True) 
        
        self.set_joint_angles(1.8, "LShoulderPitch")
        self.set_joint_angles(0.4, "LShoulderRoll")

        self.set_joint_angles(-0.4, "LElbowYaw")
        self.set_joint_angles(-0.4, "LElbowRoll")

        rospy.sleep(3.0)

        self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!


    def left_arm_repeat_move(self):

        while self.head_touch:
            print("iterating repetitive arm motion")

            #self.left_arm_home()
            
            self.set_stiffness(True)
            
            self.set_joint_angles(0.35, "LShoulderPitch")
            self.set_joint_angles(1.05, "LShoulderRoll")

            self.set_joint_angles(-1.4, "LElbowYaw")
            self.set_joint_angles(-1.4, "LElbowRoll")

            rospy.sleep(2.0)

            self.set_stiffness(False)

            self.left_arm_home()

        # self.left_arm_home()


    def central_execute(self):
        rospy.init_node('central_node',anonymous=True) #initilizes node, sets name

        # create several topic subscribers
        rospy.Subscriber("key", String, self.key_cb)
        rospy.Subscriber("joint_states",JointState,self.joints_cb)
        rospy.Subscriber("bumper",Bumper,self.bumper_cb)
        rospy.Subscriber("tactile_touch",HeadTouch,self.touch_cb)
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb)
        self.jointPub = rospy.Publisher("joint_angles",JointAnglesWithSpeed,queue_size=10)


        # test sequence to demonstrate setting joint angles
        # self.set_stiffness(True) # don't let the robot stay enabled for too long, the motors will overheat!! (don't go for lunch or something)
        # rospy.sleep(1.0)
        # self.set_joint_angles(0.5, "RShoulderPitch")
        # rospy.sleep(3.0)
        # self.set_joint_angles(0.0, "RShoulderPitch")
        # rospy.sleep(3.0)
        # self.set_stiffness(False) # always check that your robot is in a stable position before disabling the stiffness!!

        rate = rospy.Rate(10) # sets the sleep time to 10ms

        while not rospy.is_shutdown():
            self.set_stiffness(self.stiffness)

            if self.head_touch:
                self.left_arm_repeat_move()

        if data.button == 3 and not self.r_arm_touch:
            
            #self.left_arm_repeat_move()
            rate.sleep()

    # rospy.spin() just blocks the code from exiting, if you need to do any periodic tasks use the above loop
    # each Subscriber is handled in its own thread
    #rospy.spin()

if __name__=='__main__':
    # instantiate class and start loop function
    central_instance = Central()
    central_instance.central_execute()
