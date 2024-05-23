#!/usr/bin/env python3

import rospy
from TurtleBot import TurtleBot
from std_msgs.msg import Empty
from enum import Enum

class States(Enum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    RUN = 5


class handsy_bot:
    def __init__(self):
        self.state = States.STOP

        self.init_sub = rospy.Subscriber('/init', Empty, self.stop)
        self.forward_sub = rospy.Subscriber('/forward', Empty, self.forward)
        self.left_sub = rospy.Subscriber('/left', Empty, self.left)
        self.right_sub = rospy.Subscriber('/right', Empty, self.right)
        self.attack_sub = rospy.Subscriber('/attack', Empty, self.attack)
        self.run_sub = rospy.Subscriber('/run', Empty, self.run)

    def stop(self, msg):
        print("received stop")
        self.state = States.STOP

    def forward(self, msg):
        print("received forward")
        self.state = States.FORWARD

    def left(self, msg):
        self.state = States.LEFT

    def right(self, msg):
        self.state = States.RIGHT

    def attack(self, msg):
        self.state = States.ATTACK

    def run(self, msg):
        self.state = States.RUN


if __name__=='__main__':
    rospy.init_node("HandsBot")
    bot = TurtleBot()
    hands = handsy_bot()

    while not rospy.is_shutdown():
        if hands.state == States.STOP:
            bot.set_speeds(0, 0, 0)

        elif hands.state == States.FORWARD:
            bot.set_speeds(0.25, 0, 0)
        
        elif hands.state == States.LEFT:
            bot.set_speeds(0, 0, 0.25)

        elif hands.state == States.RIGHT:
            bot.set_speeds(0, 0, -0.25)

        elif hands.state == States.ATTACK:
            #figure out some sort of logic here
            pass

        elif hands.state == States.RUN:
            #figure out some sort of logic here
            pass
            
        #Publish topic 
        bot.publish_cmd_vel()

