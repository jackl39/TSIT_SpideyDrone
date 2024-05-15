#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty
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
        self.images = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu0', Imu, queue_size=1)
        self.cmd_pub= rospy.Publisher('/tello/cmd_vel', Twist, queue_size=1)
        self.image = Image()
        self.imu = Imu()

        # while self.takeoff.get_num_connections() < 1:
        #    pass
        # self.takeoff.publish(Empty())
        # rospy.on_shutdown(self.land)


    def image_republish(self, tello_image):
        self.image = tello_image
        image_time = time.time()
        image_seconds = math.floor(time.time())
        image_nanoseconds = round((image_time - image_seconds)*10**9)
        self.image.header.stamp.secs = image_seconds
        self.image.header.stamp.nsecs = image_nanoseconds
        self.images.publish(self.image)


        #Testing purposes
        self.cmd_vel_publisher()
        


    def imu_republish(self, tello_imu):
        self.imu = tello_imu
        #Inserting timestamp into image and imu
        pub_time = time.time()
        time_seconds = math.floor(time.time())
        time_nanoseconds = round((pub_time - time_seconds)*10**9)
        self.imu.header.stamp.secs = time_seconds
        self.imu.header.stamp.nsecs = time_nanoseconds
        #publishing data
        self.imu_pub.publish(self.imu)
    
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
    
    image_sub = rospy.Subscriber('/tello/camera/image_raw', Image,
                lambda msg: drone.image_republish(msg), queue_size=1)
    
    imu_sub = rospy.Subscriber('/tello/imu', Imu,
                lambda msg: drone.imu_republish(msg), queue_size=1)
    rospy.loginfo("Node has been Initialised")
    rospy.spin()
