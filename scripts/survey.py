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
from drone_republisher import droneRepublisher


# Validated and operational
# Integration


time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)
    

if __name__ == '__main__':
    
    rospy.init_node("survey")
    rospy.loginfo("Node has been Initialised")

    drone = droneRepublisher()
    while not rospy.is_shutdown():
        print(drone.get_pose())
        drone.survey_spidey

    rospy.spin()  