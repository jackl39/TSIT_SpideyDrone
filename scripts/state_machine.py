import math
import TSIT_SpideyDrone.scripts.hover_test as hover_test
import rospy
import time

distance = 15
num_y = 4
num_x = 4

time_obj = time.gmtime(0)
epoch = time.asctime(time_obj)


class stateMachine():
    def __init__(self, drone):
        '''
        Lowers drone to lowest position, finds initial position
        Once completed, next state is Move
        '''
        drone.set_speeds(0, 0, -0.25, 0, 0, 0)
        end_time = time.time() + 7

        while time.time() < end_time:
            pass
        drone.set_speeds(0, 0, 0, 0, 0, 0)
        self.init_x, self.init_y, self.init_z = drone.get_pose()
        self.next_x = self.init_x
        self.next_y = self.init_y + distance
        
        self.move(drone)

        pass


    def move(self, drone):
        '''
        Moves in y direction specified by distance
        Once moved enough, goes to Rotate state
        '''
        curr_x, curr_y, curr_z = drone.get_pose()

        while curr_y < self.next_y:
            drone.set_speeds(0, 0.25, 0, 0, 0, 0)

        drone.set_speeds(0, 0, 0, 0, 0, 0)
        self.rotate(drone)

        

    def rotate(self, drone):
        '''
        Rotate 360 degrees, pausing for 2 seconds at each 90 degree
        Next state is either move or turn (depends on which intersection at)
        '''
        finished = False
        drone.set_speeds(0, 0, 0, 0, 0, 0.1)
        while not finished:
            if 

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
    
