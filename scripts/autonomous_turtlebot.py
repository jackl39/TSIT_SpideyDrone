#!/usr/bin/env python3
import rospy
from City import City

if __name__ == '__main__':
    rospy.init_node('Localiser', anonymous=True)
    try:
        city = City()
        city.run()
    except rospy.ROSInterruptException:
        pass