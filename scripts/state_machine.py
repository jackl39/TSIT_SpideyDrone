import math
import drone_republisher as Drone
import rospy
import time

distance = 15
num_y = 4
num_x = 4

time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)


class stateMachine():
    def __init__(self):
        '''
        Lowers drone to lowest position
        Once completed, next state is Move
        '''
        pass


    def move(self):
        '''
        Moves a in y direction specified by distance
        Once moved enough, goes to Rotate state
        '''
        pass

    def rotate(self):
        '''
        Rotate 360 degrees, pausing for 2 seconds at each 90 degree
        Next state is either move or turn (depends on which intersection at)
        '''
        pass

    def turn(self):
        '''
        turns the drone for next grid 
        '''
        pass

    def end(self):
        '''
        i want to kms so this function will end me
        '''
        pass
    
