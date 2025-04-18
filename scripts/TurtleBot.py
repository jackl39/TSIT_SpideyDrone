#!/usr/bin/env python3

import numpy as np
import cv2
import math
import rospy
import tf
import apriltag
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Image

CAMERA_MATRIX = np.array([[503.038912, 0.00, 338.40326932],
                          [0.00, 499.01230583, 239.41331672],
                          [0.00, 0.00, 1.00]], dtype=np.float64) 
DIST_COEFFS = np.array([0.21411831, -0.48345064, -0.00170004, 0.02660419, 0.32653534])
# 0.055 new prints
TAG_SIZE = 0.08

class TurtleBot:

    def __init__(self):
        print("TurtleBot Initialised")
        self.detector = apriltag.Detector()
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.camera_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.camera_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, queue_size=100, callback=self.lidar_callback)
        self.camera_pub = rospy.Publisher('/spiderBot/AprilTag', Image, queue_size=1)
        
        self.odom = Odometry()
        self.command = Twist()
        self.x = 0
        self.y = 0
        self.theta = 0
        self.tag_id = None
        self.lidar_data = []
        self.translation_vector = None
        self.minDist = None
        self.distance = None

    def getTagID(self):
        return self.tag_id

    def odom_callback(self, odom_data):
        self.odom = odom_data
        self.update_pose()

    def lidar_callback(self, data):
        self.lidar_data = data.ranges
        distances = [distance for distance in data.ranges if distance > 0]
        self.minDist = min(distances)

    def camera_callback(self, cam_data):
        np_arr = np.frombuffer(cam_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)
        rotated_undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(rotated_undistorted_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags in the image
        results = self.detector.detect(gray)

        if results is not None:
            # Draw results and calculate the pose
            for result in results:
                (ptA, ptB, ptC, ptD) = result.corners
                ptA = (int(ptA[0]), int(ptA[1]))
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))

                cv2.line(rotated_undistorted_frame, ptA, ptB, (0, 255, 0), 2)
                cv2.line(rotated_undistorted_frame, ptB, ptC, (0, 255, 0), 2)
                cv2.line(rotated_undistorted_frame, ptC, ptD, (0, 255, 0), 2)
                cv2.line(rotated_undistorted_frame, ptD, ptA, (0, 255, 0), 2)

                self.tag_id = result.tag_id
                cv2.putText(rotated_undistorted_frame, f"ID: {self.tag_id}", (ptA[0], ptA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                fx, fy, cx, cy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1], CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
                pose, e0, e1 = self.detector.detection_pose(result, (fx, fy, cx, cy), TAG_SIZE)

                # Check if pose estimation is valid before proceeding
                if pose is not None and pose.shape == (4, 4):  # Ensuring the pose matrix is 4x4
                    R = pose[:3, :3]  # Extract rotation matrix
                    t = pose[:3, 3]   # Extract translation vector
                    self.distance = np.linalg.norm(t)
                    self.translation_vector = t
                    cv2.putText(rotated_undistorted_frame, f"Distance: {self.distance:.2f}m", (ptA[0], ptA[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            self.translation_vector = None
            print("No April tag detected")
        # Display the frame
        image = CvBridge().cv2_to_imgmsg(rotated_undistorted_frame, encoding="bgr8")
        self.camera_pub.publish(image)

    def update_pose(self):
        self.x = self.odom.pose.pose.position.x
        self.y = self.odom.pose.pose.position.y
        orientation = self.odom.pose.pose.orientation
        _, _, self.theta = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])

    def get_position(self):
        return self.x, self.y
    
    def get_pose(self):
        return self.theta
    
    def get_distance_to_tag(self):
        return self.distance

    def set_speeds(self, x_dot, y_dot, z_ang):
        self.command.linear.x = x_dot
        self.command.linear.y = y_dot
        self.command.angular.z = z_ang

    def publish_cmd_vel(self):
        self.cmd_vel_pub.publish(self.command)

    def rotate_to(self, target_theta):
        while abs(target_theta - self.theta) > 0.05:
            angular_difference = target_theta - self.theta
            if angular_difference > math.pi:
                angular_difference -= 2 * math.pi
            elif angular_difference < -math.pi:
                angular_difference += 2 * math.pi

            angular_speed = 0.3 * angular_difference / abs(angular_difference)
            self.set_speeds(0, 0, angular_speed)
            self.publish_cmd_vel()
            rospy.sleep(0.05)
            self.update_pose()
        self.set_speeds(0, 0, 0)
        self.publish_cmd_vel()

    def move_toward_tag(self):
        if self.translation_vector is not None:
            distance = np.linalg.norm(self.translation_vector)
            while distance > 0.6:
                x_dot = distance * 0.2
                z_ang = -0.5 * math.atan2(self.translation_vector[0], self.translation_vector[2])
                self.set_speeds(x_dot, 0, z_ang)
                self.publish_cmd_vel()
                rospy.sleep(0.05)
                self.update_pose()
                distance = np.linalg.norm(self.translation_vector)
            self.set_speeds(0, 0, 0)
            self.publish_cmd_vel()
        else:
            self.rotate_by_angle(90)

    def rotate_by_angle(self, angle):
        target_theta = self.theta + math.radians(angle)
        if target_theta > math.pi:
            target_theta -= 2 * math.pi
        elif target_theta < -math.pi:
            target_theta += 2 * math.pi
        self.rotate_to(target_theta)

    def find_streets(self):
        available_streets = []
        for angle in range(0, 360, 90):
            self.rotate_by_angle(90)
            if self.translation_vector is not None:
                available_streets.append(angle)
        return available_streets

    def avoid_collisions(self):
        pass
        # if self.minDist is not None:
        #     if self.minDist < 0.1:
        #         print("Collision avoidance")
        #         self.set_speeds(0, 0, 0)
        #         self.publish_cmd_vel()