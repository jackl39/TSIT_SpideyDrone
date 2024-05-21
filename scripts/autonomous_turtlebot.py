#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import math
import tf
import apriltag

CAMERA_MATRIX = np.array([[503.038912, 0.00, 338.40326932],
 [0.00, 499.01230583, 239.41331672],
 [0.00, 0.00, 1.00]], dtype=np.float64) 
DIST_COEFFS = np.array([0.21411831, -0.48345064, -0.00170004, 0.02660419, 0.32653534])
TAG_SIZE = 0.05

class TurtleBot:

    def __init__(self):
        rospy.init_node('turtlebot_controller', anonymous=True)
        detector = apriltag.Detector()
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.camera_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, 
                            lambda msg: self.camera_callback(msg, detector=detector))
        self.camera_pub = rospy.Publisher('/april_tag', Image, queue_size=1)
        
        self.odom = Odometry()
        self.command = Twist()
        self.x = 0
        self.y = 0
        self.theta = 0

    def odom_callback(self, odom_data):
        self.odom = odom_data
        self.update_pose()

    def camera_callback(self, cam_data, detector):
        np_arr = np.frombuffer(cam_data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)
        rotated_undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(rotated_undistorted_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags in the image
        results = detector.detect(gray)

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

                tag_id = result.tag_id
                cv2.putText(rotated_undistorted_frame, f"ID: {tag_id}", (ptA[0], ptA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                fx, fy, cx, cy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1], CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
                pose, e0, e1 = detector.detection_pose(result, (fx, fy, cx, cy), TAG_SIZE)

                # Check if pose estimation is valid before proceeding
                if pose is not None and pose.shape == (4, 4):  # Ensuring the pose matrix is 4x4
                    R = pose[:3, :3]  # Extract rotation matrix
                    t = pose[:3, 3]   # Extract translation vector
                    distance = np.linalg.norm(t)
                    cv2.putText(rotated_undistorted_frame, f"Distance: {distance:.2f}m", (ptA[0], ptA[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("No april tag detected")
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
            rospy.sleep(0.1)
            self.update_pose()
        self.set_speeds(0, 0, 0)
        self.publish_cmd_vel()

    def go_to_position(self, x, y):
        threshold = 0.05
        while math.sqrt((x - self.x)**2 + (y - self.y)**2) > threshold:
            target_theta = math.atan2(y - self.y, x - self.x)
            angular_difference = target_theta - self.theta
            if angular_difference > math.pi:
                angular_difference -= 2 * math.pi
            elif angular_difference < -math.pi:
                angular_difference += 2 * math.pi
            
            angular_speed = 0.3 * angular_difference
            distance = math.sqrt((x - self.x)**2 + (y - self.y)**2)
            linear_speed = 0.1 * distance

            self.set_speeds(linear_speed, 0, angular_speed)
            self.publish_cmd_vel()
            rospy.sleep(0.1)
            self.update_pose()
        self.set_speeds(0, 0, 0)
        self.publish_cmd_vel()

if __name__ == '__main__':
    try:
        bot = TurtleBot()
        bot.go_to_position(0.7, 0)
        bot.go_to_position(0.7, 0.6)
        bot.go_to_position(0, 0.6)
        bot.go_to_position(0, 0.9)
    except rospy.ROSInterruptException:
        pass