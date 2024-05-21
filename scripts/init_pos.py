#!/usr/bin/env python3
import rospy
import cv2
import math
import time
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
# from tello_driver.msg import TelloStatus


time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)


class droneRepublisher:
    """
    Starts off takeoff sequence and also landing sequency. Republishes image data for ORBSLAM
    """
    def __init__(self):
        # self.bridge = CvBridge()
        # self.image = Image
        # ###
        # image_sub = rospy.Subscriber('tello/image_raw', Image, 
        #         lambda msg: drone.image_subscriber, queue_size=1)

        self.takeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=1)
        self.landing = rospy.Publisher('/tello/land', Empty, queue_size=1)
        self.cmd_pub = rospy.Publisher('/tello/cmd_vel', Twist, queue_size=1)
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_z = 0.0

        self.odom = Odometry()
        odom_sub = rospy.Subscriber('/tello/odom', Odometry,
                lambda msg: drone.odom_subscriber(msg), queue_size=1)
        
        # self.time = TelloStatus()
        # time_sub = rospy.Subscriber('/tello/status/flight_time_sec', TelloStatus, 
        #         lambda msg: drone.time_subscriber(msg), queue_size=1)
        
        while self.takeoff.get_num_connections() < 1:
           pass
        self.takeoff.publish(Empty())

        rospy.on_shutdown(self.land)

    def get_pose(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        z = self.odom.pose.pose.position.z
        return x, y, z
    
    # def get_time(self): 
    #     t = self.time.flight_time_sec
    #     return t

    def get_twist(self):
        print(self.odom.pose.pose.orientation)

    def odom_subscriber(self, odom_data):
        self.odom = odom_data

    # def image_subscriber(self, img_data): 
    #     self.image = img_data

    # def time_subscriber(self, time_data):
    #     self.time = time_data

    def trans_speeds(self, x_dot, y_dot, z_dot, x_ang, y_ang, z_ang):
        command = Twist()
        command.linear.x = x_dot
        command.linear.y = y_dot
        command.linear.z = z_dot
        command.angular.x = x_ang
        command.angular.y = y_ang
        command.angular.z = z_ang
        self.cmd_pub.publish(command)

    def cmd_vel_publisher(self):
        command = Twist()
        print(command.linear.y)
        command.linear.y = -0.25
        self.cmd_pub.publish(command)

    def land(self):
        while self.landing.get_num_connections() < 1:
           pass
        self.landing.publish(Empty())

    # def colour_threshold(self): 
    #     if self.image is None: 
    #         return
        
        # hsv_image = cv2.cvtColor(self.image, cv.COLOR_BGR2HSV)
        # orange_flag = cv2.countNonZero(cv2.inRange(hsv_image, np.array([0, 100, 100]), np.array([10, 255, 255])))  # frame

        # if orange_flag > 20: 
        #     
        # else:
        #     


if __name__ == '__main__':
    rospy.init_node("drone_republisher")

    drone = droneRepublisher()
    
    rospy.loginfo("Node has been Initialised")

    start_time = time.time()
    origin = False

    #cv2.imshow("Image", self.image)

    while not rospy.is_shutdown():
        current_time = time.time()
        if not origin and (current_time - start_time) >= 7:
            drone.start_x, drone.start_y, drone.start_z = drone.get_pose()
            origin = True

        print(f"Origin: x={drone.start_x}, y={drone.start_y}, z={drone.start_z}")

        print("Pose")
        print(drone.get_pose())


        #print("twist")
        #print(drone.get_twist())
        
