#!/usr/bin/env python3

import time
import rospy
import math
from std_msgs.msg import String
from TurtleBot import TurtleBot
from Map import Map
from Intersection import Intersection, Status
import threading

# Define the City class to manage the bot and drone in a city environment
class City:
    def __init__(self):
        # Initial setup for the City class. This function initialises the ROS publishers,
        # sets up the bot, initialises the map, and subscribes to the ROS topic for
        #  gesture results.
        
        # Initialisation message
        print("City Initialised")

        # Initializing class variables
        self.bot = TurtleBot()

        # ROS subscribers & publishers
        rospy.Subscriber("/gesture_result", String, self.gesture_callback)
        self.botLocationPub = rospy.Publisher("/Spiderman/Location", String, queue_size=1)
        self.droneLocationPub = rospy.Publisher("/SpideyDrone/location", String, queue_size=1)

        # Initialize map and print it
        self.map = Map()
        self.map.print_map()

        # Direction mapping for bot movement
        self.direction_map = {
            0: "North", 1: "North", 2: "North",
            3: "East",  4: "East",  5: "East",
            6: "South", 7: "South", 8: "South",
            9: "West", 10: "West", 11: "West"
        }

        # Street naming and mapping to direction
        self.street_map = {
            0: "1st St", 8: "1st St",
            1: "2nd St", 7: "2nd St",
            2: "3rd St", 6: "3rd St",
            3: "1st Ave", 11: "1st Ave",
            4: "2nd Ave", 10: "2nd Ave",
            5: "3rd Ave", 9: "3rd Ave"
        }

        # Convert street names to numerical IDs for processing
        self.street_numbers = {
            "1st St" : 0,
            "2nd St" : 1,
            "3rd St" : 2,
            "1st Ave" : 3, 
            "2nd Ave" : 4,
            "3rd Ave" : 5
        }

        # Define intersections based on street coordinates
        self.intersections = {
            (0, 3): "First and First", (3, 0): "First and First",
            (0, 4): "First and Second", (4, 0): "First and Second",
            (0, 5): "First and Third", (5, 0): "First and Third",
            (1, 3): "Second and First", (3, 1): "Second and First",
            (1, 4): "Second and Second", (4, 1): "Second and Second",
            (1, 5): "Second and Third", (5, 1): "Second and Third",
            (2, 3): "Third and First", (3, 2): "Third and First",
            (2, 4): "Third and Second", (4, 2): "Third and Second",
            (2, 5): "Third and Third", (5, 2): "Third and Third"
        }

        # Initialize state variables for tag detection and movement tracking
        self.lastTag = None
        self.bot.direction = None
        self.street = None
        self.lastIntersection = None
        self.last2Tags = []
        self.lastTime = None

        rospy.spin()

    def gesture_callback(self, data):
        # This function is called whenever a gesture result is published.
        # It processes the gesture to determine the target intersection and controls the bot's movements
        # to navigate toward the target, handling route calculation, bot rotation, and movement.
        try:
            # Extract street1 and street2 from gesture recognition result
            streets = data.data.split(" and ")
            street1 = streets[0]
            street2 = streets[1]
            target = f"{street1} and {street2}"

            while not rospy.is_shutdown():
                available_streets = self.bot.find_streets()
                self.localize_april_tag()
                print(f"the bots direction {self.direction_map.get(self.bot.getTagID())}")
                print(f"The current intersection {self.lastIntersection}")

                route = self.map.find_shortest_path(self.lastIntersection, target)
                print(route)
                curr_inter = self.lastIntersection
                for inter in route:
                    if curr_inter == inter:
                        pass
                    elif inter != curr_inter:
                        angle = self.determine_movement(curr_inter, inter)
                        print(angle)
                        self.bot.rotate_by_angle(angle)
                        start_time = time.time()
                        while time.time() - start_time < 2:
                            pass
                        self.localize_april_tag()
                        self.bot.move_toward_tag()
                        self.bot.set_speeds(0, 0, 0)
                        self.lastIntersection = inter
                        self.bot.intersection = inter
                        curr_inter = inter
                self.success_dance()
                print("Exited Gracefully")
                return

            return
        except:
            rospy.ROSInterruptException
            pass
        finally:
            if self.drone is not None:
                self.villainFeedTransmitter.shutdown()
            else:
                ### UPDATE: CELEBRATION
                self.success_dance()
                print("Exited Gracefully")

    def success_dance(self):
        # This function makes the bot perform a celebratory dance by rotating back and forth.
        # It is typically called when the bot successfully reaches the target or when the program exits gracefully.
        self.bot.set_speeds(0, 0, 100)
        self.bot.rotate_by_angle(45)
        self.bot.rotate_by_angle(-45)
        self.bot.rotate_by_angle(45)
        self.bot.rotate_by_angle(-45)


    def localize_april_tag(self):
        # This function is responsible for detecting April tags through the bot or drone's camera.
        # It updates the bot's or drone's current position, direction, and the last two detected tags,
        # as well as printing out relevant information about the tag and location.
        if self.bot is not None:
            tag_id = self.bot.getTagID()
        elif self.drone is not None:
            tag_id = self.drone.getTagID()
        else:
            print("Neither drone or bot initialised")
        self.tagId = self.bot.getTagID()
        self.lastTag = self.bot.last_tag_id
        self.last2Tags.append(self.lastTag)
        self.last2Tags.append(self.tagId)
        self.bot.direction = self.direction_map.get(tag_id, "Unknown direction")
        self.street = self.street_map.get(tag_id, "Unknown street")
        print(f"this is current tag id: {self.tagId}")
        print(f"this is the last tage?? {self.lastTag}")
        print(f'these are the last two tags {self.last2Tags}')
        # If the tag is different from the last one, update the last 2 tags
        while len(self.last2Tags) > 2:
            self.last2Tags.pop(0)

        if len(self.last2Tags) == 2:
            # Pass the last 2 tags to the street dictionary
            # to get the street names of both tags and the intersection
            streetName1 = self.street_map.get(self.last2Tags[0], "Unknown street")
            streetName2 = self.street_map.get(self.last2Tags[1], "Unknown street")
            # Convert street names to numbers
            street1 = self.street_numbers.get(streetName1, "Unknown street")
            street2 = self.street_numbers.get(streetName2, "Unknown street")
            # print(f"Street 1: {street1} Street2: {street2}")
            intersection = self.intersections.get((street1, street2), "Unknown intersection")
            self.lastIntersection = intersection
            # print(f"Direction: {self.dire/ction}, Street: {self.street}, Intersection: {self.lastIntersection}")
        if self.lastIntersection is None:
            lastIntersectionToPrint = "No intersection visited yet"
        else:
            lastIntersectionToPrint = self.lastIntersection
        print(f"Direction: {self.bot.direction}, Street: {self.street}, Intersection: {lastIntersectionToPrint}")
        # Update positions in bot and drone so that they can be drawn in GUI
        if (self.lastIntersection is not None):
            if self.bot is not None:
                self.botLocationPub.publish(self.lastIntersection)
            else:
                self.droneLocationPub.publish(self.lastIntersection)

        
    def determine_movement(self, curr_intersection, next_intersection):
        # This function calculates the required movement direction and rotation to navigate from the current
        # intersection to the next intersection. It determines the necessary rotation based on the bot's
        # current direction and the desired direction.
        #This will return the direction required for the next 
        
        curr_coord = self.intersectionsToDraw.get(curr_intersection)
        next_coord = self.intersectionsToDraw.get(next_intersection)
        y_move = next_coord[0] - curr_coord[0]
        x_move = next_coord[1] - curr_coord[1]
        next_dir = None
        if x_move > 0:
            next_dir = 3
        elif x_move < 0:
            next_dir = 1
        elif y_move > 0:
            next_dir = 2
        elif y_move < 0:
            next_dir = 0

        curr_dir = 0
        if self.bot.direction == "North":
            curr_dir = 0
        elif self.bot.direction == "South":
            curr_dir = 2
        elif self.bot.direction == "East":
            curr_dir = 3
        elif self.bot.direction == "West":
            curr_dir = 1
        
        print(self.bot.direction)
        print(curr_dir)
        print(next_dir)

        print(f"CUrrent direction: {self.bot.direction}")
        print(f"desired direction number: {next_dir}")

        clockwise = (next_dir - curr_dir) % 4
        anticlockwise = (curr_dir - next_dir) % 4
        if clockwise < anticlockwise:
            return clockwise * 90
        else:
            return -anticlockwise * 90
        
