#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
import sys, select, termios, tty
from collections import deque

class OdometrySubscriber:
    def __init__(self):
        self.subscribed = False
        self.odom_buffer = deque(maxlen=10)  # Buffer to store the last 10 odometry coordinates
        self.indentation = 0

    def odom_callback(self, msg):
        x = round(msg.pose.pose.position.x, 4)
        y = round(msg.pose.pose.position.y, 4)
        self.odom_buffer.append((x, y))

        if len(self.odom_buffer) == 10:
            average_x = sum(coord[0] for coord in self.odom_buffer) / 10
            average_y = sum(coord[1] for coord in self.odom_buffer) / 10

            # Check if robot is stationary (within a small tolerance)
            if all(abs(coord[0] - average_x) < 0.001 and abs(coord[1] - average_y) < 0.001 for coord in self.odom_buffer):
                rospy.loginfo("{}Current Odometry Position - X: {:.4f}, Y: {:.4f}".format("    " * self.indentation, average_x, average_y))

    def toggle_odom_subscription(self):
        if self.subscribed:
            self.odom_sub.unregister()
            rospy.loginfo("{}Stopped printing odometry.".format("    " * self.indentation))
            self.subscribed = False
        else:
            self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
            rospy.loginfo("{}Printing odometry... Press 'o' to stop.".format("    " * self.indentation))
            self.subscribed = True

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__ == '__main__':
    rospy.init_node('odom_subscriber', anonymous=True)
    settings = termios.tcgetattr(sys.stdin)
    odom_subscriber = OdometrySubscriber()

    rospy.loginfo("Press 'o' to start printing odometry...")

    while not rospy.is_shutdown():
        key = getKey()
        if key == 'o':
            odom_subscriber.toggle_odom_subscription()
            odom_subscriber.indentation += 1

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)  # Restore terminal settings
