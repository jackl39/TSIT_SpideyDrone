#!/usr/bin/env python3

from Status import Status
import pygame
import sys

pygame.init()

try:
    intersection_image = pygame.image.load('tile_city.jpg').convert()
    intersection_image = pygame.transform.scale(intersection_image, (512, 512))
except Exception as e:
    print(f"Failed to load intersection_image: {e}")
    sys.exit()

class Intersection:
    def __init__(self, street1, street2):
        print("Intersection Initialised")
        self.streetsToIntersections = {
            ("1st Street", "4th Street"): "First and Fourth", ("4th Street", "1st Street"): "First and Fourth",
            ("1st Street", "5th Street"): "First and Fifth", ("5th Street", "1st Street"): "First and Fifth",
            ("1st Street", "6th Street"): "First and Sixth", ("6th Street", "1st Street"): "First and Sixth",
            ("2nd Street", "4th Street"): "Second and Fourth", ("4th Street", "2nd Street"): "Second and Fourth", 
            ("2nd Street", "5th Street"): "Second and Fifth", ("5th Street", "2nd Street"): "Second and Fifth",
            ("2nd Street", "6th Street"): "Second and Sixth", ("6th Street", "2nd Street"): "Second and Sixth",
            ("3rd Street", "4th Street"): "Third and Fourth", ("4th Street", "3rd Street"): "Third and Fourth",
            ("3rd Street", "5th Street"): "Third and Fifth", ("5th Street", "3rd Street"): "Third and Fifth",
            ("3rd Street", "6th Street"): "Third and Sixth", ("6th Street", "3rd Street"): "Third and Sixth"
        }

        self.neighbouringIntersections = {
            "First and Fourth": ["First and Fifth", "Second and Fourth"],
            "Second and Fourth": ["First and Fourth", "Third and Fourth"],
            "Third and Fourth": ["Second and Fourth", "Third and Fifth"],
            "First and Fifth": ["First and Fourth", "First and Sixth", "Second and Fifth"],
            "Second and Fifth": ["First and Fifth", "Second and Fourth", "Second and Sixth", "Third and Fifth"],
            "Third and Fifth": ["Second and Fifth", "Third and Fourth", "Third and Sixth"],
            "First and Sixth": ["First and Fifth", "Second and Sixth"],
            "Second and Sixth": ["First and Sixth", "Second and Fifth", "Third and Sixth"],
            "Third and Sixth": ["Second and Sixth", "Third and Fifth"]
        }

        self.streetNameFromIntersection = {
            "First and Fourth" : ["1st Street", "4th Street"],
            "First and Fifth" : ["1st Street", "5th Street"],
            "First and Sixth" : ["1st Street", "6th Street"],
            "Second and Fourth" : ["2nd Street", "4th Street"],
            "Second and Fifth" : ["2nd Street", "5th Street"],
            "Second and Sixth" : ["2nd Street", "6th Street"],
            "Third and Fourth" : ["3rd Street", "4th Street"],
            "Third and Fifth" : ["3rd Street", "5th Street"],
            "Third and Sixth" : ["3rd Street", "6th Street"]
        }

        self.street1 = street1
        self.street2 = street2
        self.intersectionName = self.streetsToIntersections.get((self.street1, self.street2), "Unknown intersection")
        self.neighbours = self.neighbouringIntersections.get(self.intersectionName, "Unknown intersections")
        self.status = Status() 
        self.intersection_image = intersection_image

    def update_status(self, new_status):
        if new_status in [Status.SAFE, Status.UNSAFE, Status.GOAL]:
            self.status.state = new_status