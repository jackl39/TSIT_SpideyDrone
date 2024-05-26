#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class CameraSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)  # Update topic name if needed
        self.image_pub = rospy.Publisher("/flipped_image", Image, queue_size=10)
        rospy.loginfo("Camera Subscriber Initialized")

    def image_callback(self, data):
        rospy.loginfo("Image received")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Flip the image
        flipped_image = cv2.flip(cv_image, 1)  # Flip around y-axis (horizontal flip)

        # Display the flipped image
        cv2.imshow("Flipped Camera Image", flipped_image)
        cv2.waitKey(1)

        try:
            flipped_msg = self.bridge.cv2_to_imgmsg(flipped_image, "bgr8")
            self.image_pub.publish(flipped_msg)
            rospy.loginfo("Flipped image published")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

if __name__ == '__main__':
    rospy.init_node('camera_subscriber', anonymous=True)
    cs = CameraSubscriber()
    rospy.loginfo("Camera Subscriber Node Started")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()
