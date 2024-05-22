#!/usr/bin/env python3
import rospy
import cv2 as cv
import math
import time
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# PROBLEM SUBSCRIPTION
# literally out of my asshole 

# Add Dictionary 


time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)

class droneRepublisher:

    def __init__(self):

        self.takeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=1)
        self.landing = rospy.Publisher('/tello/land', Empty, queue_size=1)
        self.cmd_pub = rospy.Publisher('/tello/cmd_vel', Twist, queue_size=1)

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

class stateMachine:

    def __init__(self):
        while self.state != "End":
            if self.state == "Forward":
                self.fly_forward(self)
            elif self.state == "Turn":
                self.turn(self)
            elif self.state == "Next Node":
                self.next_node(self)
            self.end(self)


    def measure_wall(self):
        # april tags 
        pass

    def detect_villain(self):
        # cnn
        pass

    def fly_forward(self):
        if self.measure_wall() or self.detect_villain():
            self.state = "Turn"


    def turn(self):
        turn_count = 0

        while turn_count < 4:
            # turn 90 degrees? 
            turn_count += 1
            # edit dictionary 

            if turn_count == 4:
                self.state = "Next Node"


    def next_node(self):
        nodes_travelled = 0
        villain_nodes = 0

        while (nodes_travelled + villain_nodes) < 9:
            
            # (if/else if nest)
            # mapping logic 

            if (nodes_travelled + villain_nodes) == 9:
                self.state = "End"
            else:
                self.state = "Forward"

    def end(self):
        # ???
        flag = 1 ###


if __name__ == '__main__':
    
    rospy.init_node("survey")
    rospy.loginfo("Node has been Initialised")

    drone = droneRepublisher()

    

    rospy.spin()  