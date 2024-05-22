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
        

        # while self.takeoff.get_num_connections() < 1:
        #    pass
        # self.takeoff.publish(Empty())

        # rospy.on_shutdown(self.land)

    def land(self):
        while self.landing.get_num_connections() < 1:
           pass
        self.landing.publish(Empty())
        


if __name__ == '__main__':
    rospy.init_node("drone_republisher")

    drone = droneRepublisher()
    
    rospy.loginfo("Node has been Initialised")

    rospy.spin()
