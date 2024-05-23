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
from drone_republisher import droneRepublisher


# PROBLEM SUBSCRIPTION
# NOT VALIDATED 

time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)

colours = []
inFrame = []

windowCaptureName = 'Video Capture'
windowDetectionName = 'Object Detection'

spidey = 0

    
class colourLimit:
    def __init__(self, colour, lower, upper):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/tello/camera/image_raw', Image, self.image_callback, queue_size=1)
        
        self.colour = colour
        self.lower = lower
        self.upper = upper
    
    def image_callback(self, msg):  
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            colourThreshold(cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
        except Exception as e: 
            rospy.logerr(f"Callback error: {e}")

class detectClassify:
    def __init__(self, colour, centroid):
        self.colour = colour
        self.centroid = centroid
    
    def __str__(self):
        c = 'Villain'
        return 'Detected: ' + c + ' ' + str(self.centroid)
    
def boundingBox(image, frame, colour):    
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        outer_contour = np.concatenate(contours)
        x, y, w, h = cv.boundingRect(outer_contour)

        cX = int(x + w / 2)
        cY = int(y + h / 2)
        detected = detectClassify(colour, (cX, cY))
        inFrame.append(detected)

        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
        rospy.loginfo(f"Detected: {detected}")

    return image

def checkDistance(c1, c2):
    for i in c1: 
        for j in c2: 
            if cv.norm(i-j) > 50: 
                return False
    return True

def colourThreshold(image):
    inFrame.clear()

    frameHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    red_mask = cv.inRange(frameHSV, np.array([168, 149, 61]), np.array([179, 255, 255]))  # red
    blue_mask = cv.inRange(frameHSV, np.array([100, 150, 0]), np.array([140, 255, 255]))  # blue

    red_contour, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blue_contour, _ = cv.findContours(blue_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for red in red_contour:
        for blue in blue_contour:
            if checkDistance(red, blue):
                combined = cv.bitwise_or(red_mask, blue_mask)
                image = boundingBox(image, combined, "Spidey")
                spidey = 1

    # if spidey == 1:
    #     drone.cmd_vel_publisher(0.5)
    # else:
    #     drone.cmd_vel_publisher(0.0)

    cv.imshow(windowCaptureName, image)
    cv.waitKey(1)

def search(self):
    while spidey == 0: 
        command = Twist()
        command.angular.z = -0.5
        self.cmd_pub.publish(command)
    
    red = colourLimit(0, np.array([168, 149, 61]), np.array([179, 255, 255]))   #red
    blue = colourLimit(1, np.array([100, 150, 0]), np.array([140, 255, 255]))   #blue
    colours.append(red)
    colours.append(blue)


if __name__ == '__main__':
    
    rospy.init_node("sentry")
    rospy.loginfo("Node has been Initialised")
    
    drone = droneRepublisher()
    start_time = time.time()
    origin = False
    print_once = False

    while not rospy.is_shutdown():
        current_time = time.time()
        curr_x, curr_y, curr_z = drone.get_pose()
        curr_distance = math.sqrt(curr_x**2 + curr_y**2 + curr_z**2)
        origin_distane = math.sqrt(drone.start_x**2 + drone.start_y**2 + drone.start_z**2)
        if not origin and (current_time - start_time) >= 7:
            drone.start_x, drone.start_y, drone.start_z = drone.get_pose()
            origin = True
            print_once = True
        elif print_once == True:
            print(f"Origin: x={drone.start_x}, y={drone.start_y}, z={drone.start_z}")
            print("Pose")
            print(drone.get_pose())
            print_once = False
            break 
        elif origin and curr_distance - origin_distane < 1:
            drone.trans_speeds(0.5, 0, 0, 0, 0, 0)

        elif curr_distance - origin_distane >= 1:
            drone.trans_speeds(0, 0, 0, 0, 0, 0)
            drone.land()



    rospy.sleep(5)

    rospy.spin()  