#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import apriltag
import cv2
import numpy as np
from std_msgs.msg import Empty

# Camera calibration parameters
CAMERA_MATRIX = np.array([
    [929.562627, 0.0, 487.474037],
    [0.0, 928.604856, 363.165223],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([
    [-0.016272, 0.093492, 9.3e-05, 0.002999, 0.0]
])
TAG_SIZE = 0.08

class SpideyDrone:
    def __init__(self):
        print("SpideyDrone Initialised")

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber('/tello/image_raw', Image, self.droneFeedCallback, queue_size=1) 
        self.droneFeedAprilTagPub = rospy.Publisher('/droneCam/AprilTag', Image, queue_size=1)
        self.droneFeedVillainPub = rospy.Publisher('/droneCam/Villain', Image, queue_size=1)
        self.takeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=1)
        self.landing = rospy.Publisher('/tello/land', Empty, queue_size=1)
        # Create a CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        self.lastTag = None
        self.direction = None
        self.street = None
        self.lastIntersection = None
        self.last2Tags = []
        self.lastTime = None

        while self.takeoff.get_num_connections() < 1:
           pass
        self.takeoff.publish(Empty())

        self.rawImage = None
        rospy.on_shutdown(self.land)

    def land(self):
        while self.landing.get_num_connections() < 1:
           pass
        self.landing.publish(Empty())

    def droneFeedCallback(self, data):
        try:
            detectorDrone = apriltag.Detector()

            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Detect AprilTags in the image
            results = detectorDrone.detect(gray)

            tag_detected = False
            for result in results:
                tag_detected = True
                (ptA, ptB, ptC, ptD) = result.corners
                ptA = (int(ptA[0]), int(ptA[1]))
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))

                cv2.line(cv_image, ptA, ptB, (0, 255, 0), 2)
                cv2.line(cv_image, ptB, ptC, (0, 255, 0), 2)
                cv2.line(cv_image, ptC, ptD, (0, 255, 0), 2)
                cv2.line(cv_image, ptD, ptA, (0, 255, 0), 2)

                self.tag_id = result.tag_id
                cv2.putText(cv_image, f"ID: {self.tag_id}", (ptA[0], ptA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                fx, fy, cx, cy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1], CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
                pose, e0, e1 = detectorDrone.detection_pose(result, (fx, fy, cx, cy), TAG_SIZE)

                if pose is not None and pose.shape == (4, 4):
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    self.distance = np.linalg.norm(t)
                    self.translation_vector = t
                    cv2.putText(cv_image, f"Distance: {self.distance:.2f}m", (ptA[0], ptA[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if tag_detected or len(results) == 0:
                imageAprilTag = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.droneFeedAprilTagPub.publish(imageAprilTag)
            else:
                print("No April tag detected")

            ros_image_message = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            # self.droneFeedVillainPub.publish(ros_image_message)

        except CvBridgeError as e:
            rospy.logerr("Failed to convert image: %s", e)
        except Exception as ex:
            rospy.logerr("Unexpected error: %s", ex)
