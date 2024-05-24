#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Empty
from drone_republisher import droneRepublisher
import time
import cv2 as cv 
import numpy as np
import math

colours = []
inFrame = []

windowCaptureName = 'Video Capture'
windowDetectionName = 'Object Detection'

cropped_pub = rospy.Publisher('/villain/cropped', Image, queue_size=1)
frame_pub = rospy.Publisher('villain/frames', Image, queue_size=1)
    

class colourLimit:
    def __init__(self, colour, lower, upper):
        self.bridge = CvBridge()
        
        self.image_sub = rospy.Subscriber("/camera/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1) ##
        
        self.colour = colour
        self.lower = lower
        self.upper = upper
    
    def image_callback(self, msg):  
        try: 
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            colourThreshold(cv_image) ##
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
    cropped_image = None
    
    if contours:
        # Colour Boundary
        outer_contour = np.concatenate(contours)
        x, y, w, h = cv.boundingRect(outer_contour)

        if w > 70 and h > 70: 
            cX = int(x + w / 2)
            cY = int(y + h / 2)
            detected = detectClassify(colour, (cX, cY))
            inFrame.append(detected)

            cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Villain Boundary 
            vW = int(w * 7 / 10)
            vH = int(h * 7 / 10)
            vX = int(x + (w - vW) // 2)
            vY = int(y + (h - vH) // 2)

            cv.rectangle(image, (vX, vY), (vX + vW, vY + vH), (255, 0, 0), 2)
            cropped_image = image[vY: vY + vH, vX : vX + vW]

    return image, cropped_image

def colourThreshold(image):
    inFrame.clear()

    frameHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    mask = np.zeros(frameHSV.shape[:2], dtype=np.uint8)

    if frameHSV is not None:
        for colour in colours: 
            thresholdImage = cv.inRange(frameHSV, colour.lower, colour.upper)  
            mask = cv.bitwise_or(mask, thresholdImage)
        image, cropped = boundingBox(image, mask, "Villain")   
        img_mgs = CvBridge().cv2_to_imgmsg(image, "bgr8")
        frame_pub.publish(img_mgs)

        if isinstance(cropped, np.ndarray):
            cropped_msg = CvBridge().cv2_to_imgmsg(cropped, "bgr8") 
            cropped_pub.publish(cropped_msg)


# Main
if __name__ == '__main__':

    rospy.init_node('villain_colour')

    green = colourLimit(0, np.array([40, 60, 60]), np.array([90, 190, 190]))
    # red = colourLimit(0, np.array([168, 149, 61]), np.array([179, 255, 255]))  #red
    # purple = colourLimit(1, np.array([130, 100, 180]), np.array([160, 255, 255]))  #purple 
    # pink = colourLimit(2, np.array([140, 149, 61]), np.array([168, 255, 255]))
    colours.append(green)
    # colours.append(purple)
    # colours.append(pink)
    

    rospy.loginfo("Node has been Initialised")
    rospy.spin()  