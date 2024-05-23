#!/usr/bin/env python3

from Intersection import Intersection

GRID_WIDTH, GRID_HEIGHT = 4, 4

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
        
        self.street_map = {
                    0: "1st Street", 8: "1st Street",
                    1: "2nd Street", 7: "2nd Street",
                    2: "3rd Street", 6: "3rd Street",
                    3: "4th Street", 11: "4th Street",
                    4: "5th Street", 10: "5th Street",
                    5: "6th Street", 9: "6th Street"
                }
        
        self.intersectionsLs = []
        self.grid_width = max(x for x, y in self.intersections.keys()) + 1
        self.grid_height = max(y for x, y in self.intersections.keys()) + 1
        self.grid = [[None for _ in range(self.grid_height)] for _ in range(self.grid_width)]

        for (x, y), name in self.intersections.items():
            intersection = Intersection(self.street_map.get(x, "Unknown"), self.street_map.get(y, "Unknown"))
            self.intersectionsLs.append(intersection)
            self.grid[x][y] = intersection

    def get_status(self, x, y):
        if self.grid[x][y]:
            return self.grid[x][y].safe  # Assuming Intersection class has a 'safe' attribute
        return None
    
    def getIntesection(self, intersectionName):
        for value in self.intersectionsLs:
            if value.intersection == intersectionName:
                return value
        return None

    def print_map(self):
        for value in self.intersectionsLs:
            print(f"Streets: {value.street1}, {value.street2} Intersection: {value.intersectionName} Neighbours: {value.neighbours}")
