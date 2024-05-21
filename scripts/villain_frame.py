#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
import time
import cv2 as cv 
import numpy as np
import math

colours = []
inFrame = []

windowCaptureName = 'Video Capture'
windowDetectionName = 'Object Detection'

class droneRepublisher:
    def __init__(self):
        self.takeoff = rospy.Publisher('/tello/takeoff', Empty, queue_size=1)
        self.landing = rospy.Publisher('/tello/land', Empty, queue_size=1)

        while self.takeoff.get_num_connections() < 1:
           pass
        self.takeoff.publish(Empty())

        rospy.on_shutdown(self.land)

    def land(self):
        while self.landing.get_num_connections() < 1:
           pass
        self.landing.publish(Empty())
    

class colourLimit:
    def __init__(self, colour, lower, upper):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/tello/image_raw', Image, self.image_callback, queue_size=1)
        
        self.colour = colour
        self.lower = lower
        self.upper = upper
    
    def image_callback(self, msg):  
        try: 
            rospy.loginfo("Image Received.")

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
        if self.colour == 0:
            c = 'Villain' #red 
        return 'Detected: ' + c + ' ' + str(self.centroid)
    
def boundingBox(image, frame, colour):
    maxMoments = 0
    cX = 0
    cY = 0
    x = 0
    y = 0
    w = 0
    h = 0
    
    contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] > maxMoments and M["m00"] > 3000:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            detected = detectClassify(colour, (cX, cY))
            inFrame.append(detected)
            rospy.loginfo(f"Detected {detected}")

            # Set arbitrarily -- adjust 
            x, y, w, h = cv.boundingRect(contour)
            cv.circle(image, (cX, cY), 5, (255, 255, 255), -1)
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            border = 10 
            border_x = x - border 
            border_y = y - border 
            border_w = w + 2*border
            border_h = h + 2*border 
            cv.rectangle(image, (border_x, border_y), (border_x + border_w, border_y + border_h), (255, 0, 0), 2)

    return image

def colourThreshold(image):
    rospy.loginfo("check")
    inFrame.clear()

    frameHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    if frameHSV is not None:
        for i in range(len(colours)):
            thresholdImage = cv.inRange(frameHSV, colours[i].lower, colours[i].upper)  
            image = boundingBox(image, thresholdImage, colours[i].colour)  
            if i == 5:
                cv.imshow('threshold', thresholdImage)

        cv.imshow(windowCaptureName, image)
        cv.waitKey(1)


# Main
if __name__ == '__main__':

    rospy.init_node('villain_colour')

    drone = droneRepublisher()
    red_colour = colourLimit(0, np.array([168, 149, 61]), np.array([179, 255, 255]))  #red
    colours.append(red_colour)
    
    rospy.loginfo("Node has been Initialised")
    rospy.spin()  