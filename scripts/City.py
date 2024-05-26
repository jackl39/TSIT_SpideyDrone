#!/usr/bin/env python3

import time
import rospy
import math
from std_msgs.msg import String
from TurtleBot import TurtleBot
from Map import Map, GRID_WIDTH
from SpideyDrone import SpideyDrone
from Drone2CNN import Drone2CNN
from Intersection import Intersection, Status
import pygame
import threading


WINDOW_WIDTH, WINDOW_HEIGHT = 1024, 1024
TILE_SIZE = WINDOW_WIDTH // GRID_WIDTH
WHITE = (255, 255, 255)

# Alternative separate drone and bot functionality in separate roslaunch
DEMO = "TURTLEBOT"
#DEMO = "SPIDEYDRONE"

class City:

    def __init__(self, window):
        print("City Initialised")

        self.bot = None
        self.drone = None
        self.botLocationPub = rospy.Publisher("/Spiderman/Location", String, queue_size=1)
        self.droneLocationPub = rospy.Publisher("/SpideyDrone/location", String, queue_size=1)
        if (DEMO == "TURTLEBOT"):
            self.bot = TurtleBot()
        elif (DEMO == "SPIDEYDRONE"):
            self.drone = SpideyDrone()
            self.villainFeedTransmitter = Drone2CNN()
        else:
            print("Demo type not set")
        self.map = Map()
        self.map.print_map()

        self.window = window        
        # Start the GUI in a separate thread
        self.gui_thread = threading.Thread(target=self.run_pygame)
        self.gui_thread.start()


        self.direction_map = {
            0: "North", 1: "North", 2: "North",
            3: "East",  4: "East",  5: "East",
            6: "South", 7: "South", 8: "South",
            9: "West", 10: "West", 11: "West"
        }

        self.street_map = {
            0: "1st St", 8: "1st St",
            1: "2nd St", 7: "2nd St",
            2: "3rd St", 6: "3rd St",
            3: "1st Ave", 11: "1st Ave",
            4: "2nd Ave", 10: "2nd Ave",
            5: "3rd Ave", 9: "3rd Ave"
        }

        self.street_numbers = {
            "1st St" : 0,
            "2nd St" : 1,
            "3rd St" : 2,
            "1st Ave" : 3, 
            "2nd Ave" : 4,
            "3rd Ave" : 5
        }

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

        self.intersectionsToDraw = {
            "First and First" : [0, 0],
            "First and Second" : [1, 0],
            "First and Third" : [2, 0],
            "Second and First" : [0, 1],
            "Second and Second" : [1, 1],
            "Second and Third" : [2, 1],
            "Third and First" : [0, 2],
            "Third and Second" : [1, 2],
            "Third and Third" : [2, 2]
        }

        self.lastTag = None
        self.bot.direction = None
        self.street = None
        self.lastIntersection = None
        self.last2Tags = []
        self.lastTime = None

    def run(self):
        rate = rospy.Rate(10)
        try:
            #UPDATE: GESTURE INPUT
            street1 = input("Enter street1: \n")
            street2 = input("Enter street2: \n")
            target = str(f"{street1} and {street2}")
            #target = str(input("Enter Streets: \n"))

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
                self.bot.set_speeds(0, 0, 50)
                self.bot.rotate_by_angle(45)
                self.bot.rotate_by_angle(-45)
                self.bot.rotate_by_angle(45)
                self.bot.rotate_by_angle(-45)

    def get_color_based_on_status(self, status):
        # Define colors
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        GOLD = (255, 215, 0)  # Color for 'Goal'
        GRAY = (192, 192, 192)  # Color for 'Unknown'

        # Determine the color based on the status using methods from the Status class
        if status.is_safe():
            return GREEN
        elif status.is_unsafe():
            return RED
        elif status.is_goal():
            return GOLD
        else:
            return GRAY
        
    def run_pygame(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Visalise the map and indicate each intersections status
            self.window.fill(WHITE)
            for x in range(self.map.grid_width):
                for y in range(self.map.grid_height):
                    intersection = self.map.grid[x][y]
                    if intersection:
                        self.window.blit(intersection.intersection_image, (x * TILE_SIZE, y * TILE_SIZE))
                        color = self.get_color_based_on_status(intersection.status)
                        alpha_value = 128
                        circle_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                        transparent_color = color + (alpha_value,)
                        radius = TILE_SIZE // 3
                        pygame.draw.circle(circle_surface, transparent_color, (TILE_SIZE // 2, TILE_SIZE // 2), radius)
                        self.window.blit(circle_surface, (x * TILE_SIZE, y * TILE_SIZE))

            pygame.display.flip()
            if rospy.is_shutdown():
                running = False
            pygame.time.wait(100)  # Update every 100 milliseconds
            if self.drone is not None:
                self.drone.draw(self.window, self.intersectionsToDraw.get(self.lastIntersection, None))
            elif self.bot is not None:
                self.bot.draw(self.window, self.intersectionsToDraw.get(self.lastIntersection, None))
            else:
                print("Neither drone or bot initialised")

            pygame.display.update()

        pygame.quit()

    def localize_april_tag(self):
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
        #assuming x y coordinate system with streets
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
        



# from city
# self.bot.getLocation()
# will return "First and Fourth"