#!/usr/bin/env python3

from Intersection import Intersection
import heapq
import numpy as np

GRID_WIDTH, GRID_HEIGHT = 3, 3

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
        self.safe_grid = np.zeros((self.grid_width, self.grid_height))
        print(f'printing grid {self.grid}')

        for y in range(0, 3):
            for x in range(3, 6):
                intersection = Intersection(self.street_map.get(y, "Unknown"), self.street_map.get(x, "Unknown"))
                self.intersectionsLs.append(intersection)
                self.grid[x-3][y] = intersection

    def get_status(self, x, y):
        if self.grid[x][y]:
            return self.grid[x][y].safe  # Assuming Intersection class has a 'safe' attribute
        return None
    
    def Adress2Coords(self, val):
        mydic = {
            (0, 0): "First and Fourth",
            (0, 1): "First and Fifth",
            (0, 2): "First and Sixth",
            (1, 0): "Second and Fourth",
            (1, 1): "Second and Fifth",
            (1, 2): "Second and Sixth",
            (2, 0): "Third and Fourth",
            (2, 1): "Third and Fifth",
            (2, 2): "Third and Sixth"
        }

        for key, value in mydic.items():
            if val == value:
                return key
            
    def coords2address(self, key):
        mydic = {
            (0, 0): "First and Fourth",
            (0, 1): "First and Fifth",
            (0, 2): "First and Sixth",
            (1, 0): "Second and Fourth",
            (1, 1): "Second and Fifth",
            (1, 2): "Second and Sixth",
            (2, 0): "Third and Fourth",
            (2, 1): "Third and Fifth",
            (2, 2): "Third and Sixth"
        }

        for k, value in mydic.items():
            if key == k:
                return value

    def getIntersection(self, intersectionName):
        for value in self.intersectionsLs:
            if value.intersection == intersectionName:
                return value
        return None

    def print_map(self):
        for value in self.intersectionsLs:
            print(f"Streets: {value.street1}, {value.street2} Intersection: {value.intersectionName} Neighbours: {value.neighbours}")

    def find_shortest_path(self, curr_pos, target):
        # Using a priority queue to implement Dijkstra's algorithm
        pq = []
        curr_x, curr_y = self.Adress2Coords(curr_pos)
        current_position = (curr_x, curr_y)
        heapq.heappush(pq, (0, current_position))  # (cost, position) TODO: ADD IN STARTING SPOT
        distances = {current_position: 0}
        previous_nodes = {current_position: None}

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # convert to coords
            target_x, target_y = self.Adress2Coords(target)

            if current_node == (target_x, target_y):
                break

            x, y = current_node
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring cells
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.safe_grid[nx][ny] == 0:
                    new_cost = current_distance + 1
                    if (nx, ny) not in distances or new_cost < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_cost
                        previous_nodes[(nx, ny)] = (x, y)
                        heapq.heappush(pq, (new_cost, (nx, ny)))

        # Reconstruct the path
        path = []
        step = (target_x, target_y)
        while step:
            path.append(self.coords2address(step))
            step = previous_nodes.get(step)
        path.reverse()

        return path