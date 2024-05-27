#!/usr/bin/env python3

from Status import Status
import pygame
import sys

WINDOW_WIDTH, WINDOW_HEIGHT = 1024, 1024
GRID_WIDTH, GRID_HEIGHT = 3, 3
TILE_SIZE = WINDOW_WIDTH // GRID_WIDTH

class Intersection:
    def __init__(self, street1, street2):
        print("Intersection Initialised")

        self.streetsToIntersections = {
            ("1st St", "1st Ave"): "First and First", ("1st Ave", "1st St"): "First and First",
            ("1st St", "2nd Ave"): "First and Second", ("2nd Ave", "1st St"): "First and Second",
            ("1st St", "3rd Ave"): "First and Third", ("3rd Ave", "1st St"): "First and Third",
            ("2nd St", "1st Ave"): "Second and First", ("1st Ave", "2nd St"): "Second and First", 
            ("2nd St", "2nd Ave"): "Second and Second", ("2nd Ave", "2nd St"): "Second and Second",
            ("2nd St", "3rd Ave"): "Second and Third", ("3rd Ave", "2nd St"): "Second and Third",
            ("3rd St", "1st Ave"): "Third and First", ("1st Ave", "3rd St"): "Third and First",
            ("3rd St", "2nd Ave"): "Third and Second", ("2nd Ave", "3rd St"): "Third and Second",
            ("3rd St", "3rd Ave"): "Third and Third", ("3rd Ave", "3rd St"): "Third and Third"
        }

        self.neighbouringIntersections = {
            "First and First": ["First and Second", "Second and First"],
            "Second and First": ["First and First", "Third and First"],
            "Third and First": ["Second and First", "Third and Second"],
            "First and Second": ["First and First", "First and Third", "Second and Second"],
            "Second and Second": ["First and Second", "Second and First", "Second and Third", "Third and Second"],
            "Third and Second": ["Second and Second", "Third and First", "Third and Third"],
            "First and Third": ["First and Second", "Second and Third"],
            "Second and Third": ["First and Third", "Second and Second", "Third and Third"],
            "Third and Third": ["Second and Third", "Third and Second"]
        }

        self.streetNameFromIntersection = {
            "First and First" : ["1st St", "1st Ave"],
            "First and Second" : ["1st St", "2nd Ave"],
            "First and Third" : ["1st St", "3rd Ave"],
            "Second and First" : ["2nd St", "1st Ave"],
            "Second and Second" : ["2nd St", "2nd Ave"],
            "Second and Third" : ["2nd St", "3rd Ave"],
            "Third and First" : ["3rd St", "1st Ave"],
            "Third and Second" : ["3rd St", "2nd Ave"],
            "Third and Third" : ["3rd St", "3rd Ave"]
        }

        self.street1 = street1
        self.street2 = street2
        self.intersectionName = self.streetsToIntersections.get((self.street1, self.street2), "Unknown intersection")
        self.neighbours = self.neighbouringIntersections.get(self.intersectionName, "Unknown intersections")
        self.status = Status() 

    def update_status(self, new_status):
        if new_status in [Status.SAFE, Status.UNSAFE, Status.GOAL]:
            self.status.state = new_status