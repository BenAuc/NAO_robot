#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point, PolygonStamped
from jsk_recognition_msgs.msg import PolygonArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv2.aruco as aruco
import numpy as np
import copy


class ObjectTracker:
    """
    Class implementing the tracking of an object in NAO's camera stream + ArUco markers
    @Inputs:
    -NAO's camera stream
    @Outputs:
    -coordinates of the object
    """

    def __init__(self):

        # initialize class variables
        self.blob_coordinates = None # coordinates of tracked object
        self.cam_y_max = 240 - 1 # camera resolution
        self.cam_x_max = 320 - 1 # camera resolution
        self.lower_red = np.array([161,155,84]) # lower bound of the tracked object's color in HSV
        self.upper_red = np.array([179,255,255]) # upper bound of the tracked object's color in HSV

        # define frequency of execution of this node
        self.frequency = 5.0 # Hz
        self.rate = rospy.Rate(self.frequency) # timing object

        # define subscribers
        rospy.Subscriber("/nao_robot/camera/top/camera/image_raw",Image,self.image_cb) # subscriber to NAO's camera stream

        # define topic publishers
        #self.markerPub = rospy.Publisher("/nao_robot/markers/polygon", PolygonStamped, queue_size=1)  # the polygon describing the outline of a marker + the marker ID
        self.markerListPub = rospy.Publisher("/nao_robot/markerlist", PolygonArray, queue_size=1)  # the polygon describing the outline of a marker + the marker ID

        # define ArUco parameters (source: https://people.eng.unimelb.edu.au/pbeuchat/asclinic/software/workflow_aruco_detection.html)
        self.MARKER_SIZE = 0.018 # Marker size in meters
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL) # Get the ArUco dictionary to use
        self.aruco_params = aruco.DetectorParameters_create() # Create an parameter structure needed for the ArUco detection

        # Variable to store corners and ids of ArUco markers
        self.marker_corners = None
        self.marker_ids = None


    def image_cb(self,data):
        """
        Handles incoming data from NAO's camera stream and converts the image to cv2 format
        Inputs:
        -image coming from camera stream
        Outputs:
        - calls the marker detection method self.marker_detection(cv_image)
        """

        # declare instance of CVBrige for image conversion
        bridge_instance = CvBridge()

        # try convert the image
        try:
            cv_image = bridge_instance.imgmsg_to_cv2(data,"bgr8")

        except CvBridgeError as e:
            rospy.logerr(e)
        
        # try detect the object
        try:
            self.marker_corners, self.marker_ids = self.marker_detection(cv_image)

        except:
            pass

        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly


    def marker_detection(self, image):
        """
        Detect ArUco markers
        Sources: 
        https://gitlab.unimelb.edu.au/asclinic/asclinic-system/-/blob/master/catkin_ws/src/asclinic_pkg/src/nodes/template_aruco_capture.py
        https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/

        IMPORTANT NOTE: This function allows to estimate the corners of each marker, but NOT the actual pose as [tvec, rvec], 
        where tvec is the translation vector [x, y, z] and rvec is the 3 dimensional rotation vector as an "axis angle" representation.
        To obtain this, we need to estimate the camera calibratin and distortion matrices, which can be done following this tutorial:
        https://docs.opencv.org/3.1.0/da/d13/tutorial_aruco_calibration.html
        and then compute the rvec and tvec values with
        https://aliyasineser.medium.com/aruco-marker-tracking-with-opencv-8cb844c26628
        """
        
        # Detect ArUco markers from the image
        (corners, ids, rejected) = aruco.detectMarkers(image, self.aruco_dict, parameters=self.aruco_params)

        # Process any ArUco markers that were detected
        if ids is not None:
            # Display the number of markers found
            # print("Number of ArUco markers found:" + str(len(ids)) )           

            # Flatten ArUco IDs list to make it easier to work with
            ids = ids.flatten()

            # Iterate over the markers detected
            for i_marker_id in range(len(ids)):
                # Get the ID for this marker
                this_id = ids[i_marker_id]
                # Get the corners for this marker
                corners_of_this_marker = corners[i_marker_id].reshape((4, 2))
                
                # Estimate the pose of this marker
                (topLeft, topRight, bottomRight, bottomLeft) = corners_of_this_marker
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the
                # ArUco marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
                cv2.putText(image, str(this_id),
                    (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        else:
            corners = None
            ids = None

        # Display the camera image
        cv2.imshow("Marker detection", image)

        return corners, ids


    def publish_marker_polygons(self):
        """
        This function is meant to publish the polygon shape of the ArUco markers.
        Inputs:
        -corners: 4 x 2 array of (x,y) pixel coordinates
        -ids: ArUco markers ids
        Outputs:
        -publish a list of arrays of markers ids and (x,y) pixel coordinates as a PolygonArray() message
        """

        # Process any ArUco markers that were detected
        if self.marker_ids is not None:

            # create lists of polygon arrays
            polygon_list_msg = PolygonArray()

            # Flatten ArUco IDs list to make it easier to work with
            ids = copy.deepcopy(self.marker_ids.flatten())

            # Iterate over the markers detected
            for i_marker_id in range(len(ids)):
                # Get the ID for this marker
                this_id = ids[i_marker_id]
                # Get the corners for this marker
                corners_of_this_marker = self.marker_corners[i_marker_id].reshape((4, 2))

                # Create and setup message
                ps_message = PolygonStamped()
                ps_message.header.frame_id = str(this_id)
                ps_message.polygon.points = [Point(x=corners_of_this_marker[0, 0], y=corners_of_this_marker[0, 1]),
                Point(x=corners_of_this_marker[1, 0], y=corners_of_this_marker[1, 1]),
                Point(x=corners_of_this_marker[2, 0], y=corners_of_this_marker[2, 1]),
                Point(x=corners_of_this_marker[3, 0], y=corners_of_this_marker[3, 1])]

                polygon_list_msg.polygons.append(ps_message)

                # # Publish message
                # self.markerPub.publish(ps_message)

            # Publish message
            self.markerListPub.publish(polygon_list_msg)

        else:

            # create lists of polygon arrays
            polygon_list_msg = PolygonArray()

            # Create empty message to indicate no marker is being tracked
            ps_message = PolygonStamped()
            ps_message.header.frame_id = str(0)
            ps_message.polygon.points = [Point(x=0, y=0),
            Point(x=0, y=0),
            Point(x=0, y=0),
            Point(x=0, y=0)]

            polygon_list_msg.polygons.append(ps_message)

            # Publish message
            # self.markerPub.publish(ps_message)
            self.markerListPub.publish(polygon_list_msg)
        

    def run(self):
        """
        Main loop of class.
        Inputs:
        -self
        Outputs:
        -runs the step function.
        """

        while not rospy.is_shutdown():

            # perform step
            self.step()

            # sleep to target frequency
            self.rate.sleep()


    def step(self):
        """
        Perform an iteration of marker tracking.
        Inputs:
        -self
        Outputs:
        -publish the marker coordinates as PolygonStamped() message.
        """

        # check if markers have been detected
        if self.marker_ids is not None:

            # Publish the polygons of the edges of the markers and the ids
            self.publish_marker_polygons()


if __name__=='__main__':

    #initialize the node and set name
    rospy.init_node('object_tracker',anonymous=True) #initilizes node

    # instantiate class and start loop function
    try:
        object_tracker = ObjectTracker()
        object_tracker.run()
        
    except rospy.ROSInterruptException:
        pass
