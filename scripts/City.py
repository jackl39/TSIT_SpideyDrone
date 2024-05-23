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
        self
        if (DEMO == "TURTLEBOT"):
            self.bot = TurtleBot()
        elif (DEMO == "SPIDEYDRONE"):
            self.drone = SpideyDrone()
            self.villainFeedTransmitter = Drone2CNN()
        else:
            print("Demo type not set")
        self.map = Map()
        # self.map.print_map()

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
            0: "1st Street", 8: "1st Street",
            1: "2nd Street", 7: "2nd Street",
            2: "3rd Street", 6: "3rd Street",
            3: "4th Street", 11: "4th Street",
            4: "5th Street", 10: "5th Street",
            5: "6th Street", 9: "6th Street"
        }

        self.street_numbers = {
            "1st Street" : 0,
            "2nd Street" : 1,
            "3rd Street" : 2,
            "4th Street" : 3, 
            "5th Street" : 4,
            "6th Street" : 5
        }

        self.intersections = {
            (0, 3): "First and Fourth", (3, 0): "First and Fourth",
            (0, 4): "First and Fifth", (4, 0): "First and Fifth",
            (0, 5): "First and Sixth", (5, 0): "First and Sixth",
            (1, 3): "Second and Fourth", (3, 1): "Second and Fourth",
            (1, 4): "Second and Fifth", (4, 1): "Second and Fifth",
            (1, 5): "Second and Sixth", (5, 1): "Second and Sixth",
            (2, 3): "Third and Fourth", (3, 2): "Third and Fourth",
            (2, 4): "Third and Fifth", (4, 2): "Third and Fifth",
            (2, 5): "Third and Sixth", (5, 2): "Third and Sixth"
        }

        self.intersectionsToDraw = {
            "First and Fourth" : [0, 0],
            "First and Fifth" : [1, 0],
            "First and Sixth" : [2, 0],
            "Second and Fourth" : [0, 1],
            "Second and Fifth" : [1, 1],
            "Second and Sixth" : [2, 1],
            "Third and Fourth" : [0, 2],
            "Third and Fifth" : [1, 2],
            "Third and Sixth" : [2, 2]
        }

        self.lastTag = None
        self.direction = None
        self.street = None
        self.lastIntersection = None
        self.last2Tags = []
        self.lastTime = None

    def run(self):
        rate = rospy.Rate(10)
        try:
            while not rospy.is_shutdown():
                self.localize_april_tag()
                # self.bot.avoid_collisions()
                available_streets = self.bot.find_streets()
                print("Available streets: ", available_streets)
                # selected_street = inputs: ", available_streets)
                selected_street = input("Select a street angle (0, 90, 180, etc.): ")
                self.bot.rotate_to(math.radians(float(selected_street)))
                # self.bot.move_to(s)
                # if (self.bot.translation_vector is not None) and (self.bot.get_distance_to_tag() > 0.7):
                #     self.bot.move_toward_tag()
                # # else (self.bot.translation_vector is not None) and (self.distance < 0.7):
                # else:
                #     self.bot.rotate_by_angle(90)
                rate.sleep()
        except:
            rospy.ROSInterruptException
            pass
        finally:
            if self.drone is not None:
                self.villainFeedTransmitter.shutdown()
            else:
                print("Exited Gracefully")

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
        self.tagId = tag_id
        self.direction = self.direction_map.get(tag_id, "Unknown direction")
        self.street = self.street_map.get(tag_id, "Unknown street")

        # If the tag is different from the last one, update the last 2 tags
        if (tag_id != self.lastTag and tag_id not in self.last2Tags):
            self.last2Tags.append(tag_id)
            # Keep only the last 2 tags
            if len(self.last2Tags) > 2:
                self.last2Tags.pop(0)
            # If the last tag was detected more than 5 seconds ago, remove it
            # This is to avoid having the same tag twice in the last 2 tags
            # and to allow time to make a turn onto a new street
            if self.lastTime is not None:
                if time.time() - self.lastTime > 20 and len(self.last2Tags) == 2:
                    self.last2Tags.pop(0)
            self.lastTag = tag_id
            self.lastTime = time.time()

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
        print(f"Direction: {self.direction}, Street: {self.street}, Intersection: {lastIntersectionToPrint}")
        # Update positions in bot and drone so that they can be drawn in GUI
        if (self.lastIntersection is not None):
            if self.bot is not None:
                self.botLocationPub.publish(self.lastIntersection)
            else:
                self.droneLocationPub.publish(self.lastIntersection)

# from city
# self.bot.getLocation()
# will return "First and Fourth"