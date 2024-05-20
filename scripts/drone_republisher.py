#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import time


time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)


class droneRepublisher:
    """
    Starts off takeoff sequence and also landing sequency. Republishes image data for ORBSLAM
    """
    def __init__(self):
        self.takeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=1)
        self.landing = rospy.Publisher('/tello/land', Empty, queue_size=1)
        self.cmd_pub= rospy.Publisher('/tello/cmd_vel', Twist, queue_size=1)
        self.odom = Odometry()

        odom_sub = rospy.Subscriber('/tello/odom', Odometry,
                lambda msg: drone.odom_subscriber(msg), queue_size=1)
        

        while self.takeoff.get_num_connections() < 1:
           pass
        self.takeoff.publish(Empty())

        rospy.on_shutdown(self.land)

    def get_pose(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        z = self.odom.pose.pose.position.z
        return x, y, z

    def get_twist(self):
        print(self.odom.pose.pose.orientation)

    def odom_subscriber(self, odom_data):
        self.odom = odom_data

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
        




if __name__ == '__main__':
    rospy.init_node("drone_republisher")

    drone = droneRepublisher()
    
    rospy.loginfo("Node has been Initialised")

    while not rospy.is_shutdown():
        print("Pose")
        drone.get_pose()
        print("twist")
        drone.get_twist()
