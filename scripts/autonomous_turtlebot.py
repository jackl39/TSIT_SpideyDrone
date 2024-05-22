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
import time

CAMERA_MATRIX = np.array([[503.038912, 0.00, 338.40326932],
 [0.00, 499.01230583, 239.41331672],
 [0.00, 0.00, 1.00]], dtype=np.float64) 
DIST_COEFFS = np.array([0.21411831, -0.48345064, -0.00170004, 0.02660419, 0.32653534])
TAG_SIZE = 0.055

class TurtleBot:

    def __init__(self):
        print("TurtleBot Initialised")
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
        self.tag_id = None

    def getTagID(self):
        return self.tag_id

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

                self.tag_id = result.tag_id
                cv2.putText(rotated_undistorted_frame, f"ID: {self.tag_id}", (ptA[0], ptA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

class City:

    def __init__(self):
        print("City Initialised")
        self.bot = TurtleBot()
        # Define the direction mapping for AprilTags
        self.direction_map = {
            0: "North", 1: "North", 2: "North",
            3: "East",  4: "East",  5: "East",
            6: "South", 7: "South", 8: "South",
            9: "West", 10: "West", 11: "West"
        }

        # Define the street mapping for AprilTags
        self.street_map = {
            0: "1st Street", 8: "1st Street",
            1: "2nd Street", 7: "2nd Street",
            2: "3rd Street", 6: "3rd Street",
            3: "4th Street", 11: "4th Street",
            4: "5th Street", 10: "5th Street",
            5: "6th Street", 9: "6th Street"
        }

        self.street_numbers = {
            "1st Street" : 0,
            "2nd Street" : 1,
            "3rd Street" : 2,
            "4th Street" : 3, 
            "5th Street" : 4,
            "6th Street" : 5
        }

        # Intersections dictionary about where you are based on the last 2 different April Tags
        self.intersections = {
            (0, 3): "First and Fourth", (3, 0): "First and Fourth",
            (0, 4): "First and Fifth", (4, 0): "First and Fifth",
            (0, 5): "First and Sixth", (5, 0): "First and Sixth",
            (1, 3): "Second and Fourth", (3, 1): "Second and Fourth",
            (1, 4): "Second and Fifth", (4, 1): "Second and Fifth",
            (1, 5): "Second and Sixth", (5, 1): "Second and Sixth",
            (2, 3): "Third and Fourth", (3, 2): "Third and Fourth",
            (2, 4): "Third and Fifth", (4, 2): "Third and Fifth",
            (2, 5): "Third and Sixth", (5, 2): "Third and Sixth"
        }
        self.lastTag = None
        self.direction = None
        self.street = None
        self.lastIntersection = None
        self.last2Tags = []
        self.lastTime = None

    def localize_april_tag(self):
        tag_id = self.bot.getTagID()
        self.tagId = tag_id
        self.direction = self.direction_map.get(tag_id, "Unknown direction")
        self.street = self.street_map.get(tag_id, "Unknown street")

        # If the tag is different from the last one, update the last 2 tags
        if (tag_id != self.lastTag and tag_id not in self.last2Tags):
            self.last2Tags.append(tag_id)
            # Keep only the last 2 tags
            if len(self.last2Tags) > 2:
                self.last2Tags.pop(0)
            # If the last tag was detected more than 5 seconds ago, remove it
            # This is to avoid having the same tag twice in the last 2 tags
            # and to allow time to make a turn onto a new street
            if self.lastTime is not None:
                if time.time() - self.lastTime > 5 and len(self.last2Tags) == 2:
                    self.last2Tags.pop(0)
            self.lastTag = tag_id
            self.lastTime = time.time()

        if len(self.last2Tags) == 2:
            # Pass the last 2 tags to the street dictionary
            # to get the street names of both tags and the intersection
            streetName1 = self.street_map.get(self.last2Tags[0], "Unknown street")
            streetName2 = self.street_map.get(self.last2Tags[1], "Unknown street")
            # Convert street names to numbers
            street1 = self.street_numbers.get(streetName1, "Unknown street")
            street2 = self.street_numbers.get(streetName2, "Unknown street")
            # print(f"Street 1: {street1} Street2: {street2}")
            intersection = self.intersections.get((street1, street2), "Unknown intersection")
            self.lastIntersection = intersection
            print(f"Direction: {self.direction}, Street: {self.street}, Intersection: {self.lastIntersection}")
        if self.lastIntersection is None:
            self.lastIntersection = "No intersection visited yet"
        print(f"Direction: {self.direction}, Street: {self.street}, Intersection: {self.lastIntersection}")

if __name__ == '__main__':
    rospy.init_node('Localiser', anonymous=True)
    try:
        city = City()
        while not rospy.is_shutdown():
            city.localize_april_tag()
            # city.bot.go_to_position(0.7, 0)
            # city.bot.go_to_position(0.7, 0.6)
            # city.bot.go_to_position(0, 0.6)
            # city.bot.go_to_position(0, 0.9)
            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        pass