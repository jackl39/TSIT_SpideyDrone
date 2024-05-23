#!/usr/bin/env python3
import rospy
import pygame
from City import City

def initialize_pygame():
    pygame.init()
    window_size = (1024, 1024)
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("City Visualization")
    return window

if __name__ == '__main__':
    rospy.init_node('Localiser', anonymous=True)
    try:
        window = initialize_pygame()
        city = City(window)
        city.run()
    except rospy.ROSInterruptException:
        pass