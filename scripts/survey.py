#!/usr/bin/env python3
import rospy
import cv2 as cv
import math
import time
import numpy as np
from std_msgs.msg import Empty
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# Validated and operational
# Integration


time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)

class droneRepublisher:

    def __init__(self):

        self.takeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=1)
        self.landing = rospy.Publisher('/tello/land', Empty, queue_size=1)
        self.cmd_pub = rospy.Publisher('/tello/cmd_vel', Twist, queue_size=1)

        self.odom = Odometry()
        odom_sub = rospy.Subscriber('/tello/odom', Odometry, self.odom_subscriber, queue_size=1)
        
        while self.takeoff.get_num_connections() < 1:
           pass
        self.takeoff.publish(Empty())

        rospy.on_shutdown(self.land)

        rospy.sleep(5) ##### 
        # fly to corner node 

        self.survey_spidey()

    def get_pose(self):
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        z = self.odom.pose.pose.position.z
        return x, y, z

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

    def survey_spidey(self):
        r = 0.5
        linear_speed = 0.5
        angular_speed = linear_speed/r

        while not rospy.is_shutdown():
            self.trans_speeds(0, linear_speed, 0, 0, 0, -angular_speed)

    def land(self):
        while self.landing.get_num_connections() < 1:
           pass
        self.landing.publish(Empty())
    

if __name__ == '__main__':
    
    rospy.init_node("survey")
    rospy.loginfo("Node has been Initialised")

    drone = droneRepublisher()
    while not rospy.is_shutdown():
        print(drone.get_pose())

    rospy.spin()  