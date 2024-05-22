#!/usr/bin/env python3

from Intersection import Intersection

class Map:
    def __init__(self):
        print("Map Initialised")
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
        self.intersectionsLs = []
        street1num = 0
        street2num = 3
        for street1num in range(0, 3):
            for street2num in range(3, 6):
                self.street_map = {
                    0: "1st Street", 8: "1st Street",
                    1: "2nd Street", 7: "2nd Street",
                    2: "3rd Street", 6: "3rd Street",
                    3: "4th Street", 11: "4th Street",
                    4: "5th Street", 10: "5th Street",
                    5: "6th Street", 9: "6th Street"
                }
                
                street1 = self.street_map.get(street1num, "Unknown street")
                street2 = self.street_map.get(street2num, "Unknown street")
                intersectionMap = Intersection(street1, street2)
                self.intersectionsLs.append(intersectionMap)
            street2num = 0

    def getIntesection(self, intersectionName):
        for value in self.intersectionsLs:
            if value.intersection == intersectionName:
                return value
        return None


    def print_map(self):
        for value in self.intersectionsLs:
            print(f"Streets: {value.street1}, {value.street2} Intersection: {value.intersectionName} Neighbours: {value.neighbours}")
