#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point, PolygonStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv2.aruco as aruco
import numpy as np


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
        self.redBlobPub = rospy.Publisher("/nao_robot/tracked_object/coordinates", Point, queue_size=1)  # the centroid coordinates of the tracked object
        self.markerPub = rospy.Publisher("/nao_robot/markers/polygon", PolygonStamped, queue_size=1)  # the polygon describing the outline of a marker + the marker ID

        # define messages
        self.blob_coordinates_msg = Point()

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
        - calls the object detection method self.object_detection(cv_image)
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
            self.object_detection(cv_image)
            self.marker_corners, self.marker_ids = self.marker_detection(cv_image)

        except:
            pass

        cv2.waitKey(3) # a small wait time is needed for the image to be displayed correctly


    def object_detection(self,image):
        """
        Extract coordinates of object of the target color in the visual field.
        Inputs:
        -image in cv2 format
        Outputs:
        -saves the object coordinates as array of (x,y) coordinates in the class variable self.blob_coordinates
        """

        # Transform image into HSV, select parts within the predefined red range color as a mask,
        # dilate and erode the selected parts to remove noise
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        kernel = np.ones((5,5),np.uint8)
        mask_dilation = cv2.dilate(mask, kernel, iterations=2)
        mask_final = cv2.erode(mask_dilation, kernel, iterations=1)
        kernel = np.ones((6,6),np.float32)/25
        mask_final = cv2.filter2D(mask_final,-1,kernel)

        # Apply mask to original image, show results
        res = cv2.bitwise_and(image,image, mask= mask_final)

        # Parameter definition for SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea  = True
        params.minArea = 1000
        params.maxArea = 200000
        params.filterByInertia = True
        params.minInertiaRatio = 0.0
        params.maxInertiaRatio  = 0.8

        # Applying the params
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(~mask_final)

        #draw 
        im_with_keypoints = cv2.drawKeypoints(~mask_final, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("Keypoints", im_with_keypoints)

        ## Find outer contours 
        im, contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maxContour = 0
        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContour:
                maxContour = contourSize
                maxContourData = contour
               
        ## Draw
        cv2.drawContours(image, maxContourData, -1, (0,255,0), 2, lineType = cv2.LINE_4)

        # Calculate image moments of the detected contour
        M = cv2.moments(maxContourData)

        try:
        # Draw a circle based centered at centroid coordinates
            cv2.circle(image, (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])), 5, (0, 0, 0), -1)

        # Show image:
            cv2.imshow("outline contour & centroid", image)

        except ZeroDivisionError:
            pass

        # Save center coordinates of the blob as a Point() message
        point_x = int(M['m10'] / M['m00'])
        point_y = int(M['m01'] / M['m00'])

        # if the coordinates of a blob could be resolved
        if point_x > 0 and point_y > 0:
            # save in class variable
            self.blob_coordinates = [point_x, point_y]

        else:
            # keep the arm steady where it was before that moment
            self.blob_coordinates = None


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


    def publish_marker_polygons(self, corners, ids):
        """
        This function is meant to show how to publish the polygon shape of the ArUco markers.
        """


        # Process any ArUco markers that were detected
        if ids is not None:          

            # Flatten ArUco IDs list to make it easier to work with
            ids = ids.flatten()

            # Iterate over the markers detected
            for i_marker_id in range(len(ids)):
                # Get the ID for this marker
                this_id = ids[i_marker_id]
                # Get the corners for this marker
                corners_of_this_marker = corners[i_marker_id].reshape((4, 2))

                # Create and setup message
                ps_message = PolygonStamped()
                ps_message.header.frame_id = str(this_id)
                ps_message.polygon.points = [Point(x=corners_of_this_marker[0, 0], y=corners_of_this_marker[0, 1]),
                Point(x=corners_of_this_marker[1, 0], y=corners_of_this_marker[1, 1]),
                Point(x=corners_of_this_marker[2, 0], y=corners_of_this_marker[2, 1]),
                Point(x=corners_of_this_marker[3, 0], y=corners_of_this_marker[3, 1])]

                # Publish message
                self.markerPub.publish(ps_message)

        else:
            # Create empty message to indicate no marker is being tracked
            ps_message = PolygonStamped()
            ps_message.header.frame_id = str(0)
            ps_message.polygon.points = [Point(x=0, y=0),
            Point(x=0, y=0),
            Point(x=0, y=0),
            Point(x=0, y=0)]

            # Publish message
            self.markerPub.publish(ps_message)
        

    def input_normalization(self, coordinates):
        """
        Normalize the blob coordinates.
        Inputs:
        -array of (x,y) coordinates of an object in the visual field.
        Outputs:
        -numpy array of normalized (x,y) coordinates.
        """

        return np.array(coordinates, dtype=float) / np.array([self.cam_x_max, self.cam_y_max], dtype=float)


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
        Perform an iteration of blob tracking.
        Inputs:
        -self
        Outputs:
        -publishes the blob coordinates as Point() message.
        """

        # Publish the polygons of the edges of the markers and the ids to ROS
        self.publish_marker_polygons(self.marker_corners, self.marker_ids)

        if self.blob_coordinates is not None:
            
            # normalize the blob coordinates
            data = self.input_normalization(self.blob_coordinates)

            # create message
            self.blob_coordinates_msg = Point(data[0], data[1], 0)

            # Publish blob coordinates
            self.redBlobPub.publish(self.blob_coordinates_msg)

            # print("object tracker node publishing coordinates : ", data)

        else:
            
            # create empty message to indicate no object is being tracked
            self.blob_coordinates_msg = Point(0, 0, 0)

            # Publish blob coordinates
            self.redBlobPub.publish(self.blob_coordinates_msg)

            print("object tracker node publishing no coordinates")


if __name__=='__main__':

    #initialize the node and set name
    rospy.init_node('object_tracker',anonymous=True) #initilizes node

    # instantiate class and start loop function
    try:
        object_tracker = ObjectTracker()
        object_tracker.run()
        
    except rospy.ROSInterruptException:
        pass
