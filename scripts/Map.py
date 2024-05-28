#!/usr/bin/env python3

from Intersection import Intersection
import heapq
import numpy as np

class Map:
    def __init__(self):
        # Initialises the Map class. Sets up the intersection grid and other class variables.
        # Also initialises intersections based on predefined coordinates.
        
        print("Map Initialised")
        
        # predefined intersections with coordinates
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
        
        # mapping of indexes to street names
        self.street_map = {
            0: "1st St", 8: "1st St",
            1: "2nd St", 7: "2nd St",
            2: "3rd St", 6: "3rd St",
            3: "1st Ave", 11: "1st Ave",
            4: "2nd Ave", 10: "2nd Ave",
            5: "3rd Ave", 9: "3rd Ave"
        }

        # Populate the grid with Intersection objects based on street mapping
        self.intersectionsLs = []
        self.grid_width = max(x for x, y in self.intersections.keys()) + 1
        self.grid_height = max(y for x, y in self.intersections.keys()) + 1
        self.grid = [[None for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self.safe_grid = np.zeros((self.grid_width, self.grid_height))

        for y in range(0, 3):
            for x in range(3, 6):
                intersection = Intersection(self.street_map.get(y, "Unknown"), self.street_map.get(x, "Unknown"))
                self.intersectionsLs.append(intersection)
                self.grid[x-3][y] = intersection

    def get_status(self, x, y):
        # Returns the safety status of an intersection at coordinates (x, y
        if self.grid[x][y]:
            return self.grid[x][y].safe  # Assuming Intersection class has a 'safe' attribute
        return None
    
    def Adress2Coords(self, val):
        # Converts intersection address to grid coordinates.
        mydic = {
            (0, 0): "First and First",
            (0, 1): "First and Second",
            (0, 2): "First and Third",
            (1, 0): "Second and First",
            (1, 1): "Second and Second",
            (1, 2): "Second and Third",
            (2, 0): "Third and First",
            (2, 1): "Third and Second",
            (2, 2): "Third and Third"
        }

        for key, value in mydic.items():
            if val == value:
                return key
            
    def coords2address(self, key):
        # Converts grid coordinates back to intersection address.
        mydic = {
            (0, 0): "First and First",
            (0, 1): "First and Second",
            (0, 2): "First and Third",
            (1, 0): "Second and First",
            (1, 1): "Second and Second",
            (1, 2): "Second and Third",
            (2, 0): "Third and First",
            (2, 1): "Third and Second",
            (2, 2): "Third and Third"
        }

        for k, value in mydic.items():
            if key == k:
                return value

    def getIntersection(self, intersectionName):
        # Retrieves an Intersection object by its name
        for value in self.intersectionsLs:
            if value.intersection == intersectionName:
                return value
        return None

    def print_map(self):
        # Prints out details of all intersections managed by this map instance.
        for value in self.intersectionsLs:
            print(f"Streets: {value.street1}, {value.street2} Intersection: {value.intersectionName} Neighbours: {value.neighbours}")

    def find_shortest_path(self, curr_pos, target):
        # Implements Dijkstra's algorithm to find the shortest path from current position to target.
        # Uses a priority queue to manage the open set of nodes to explore
        pq = []
        curr_x, curr_y = self.Adress2Coords(curr_pos)
        current_position = (curr_x, curr_y)
        heapq.heappush(pq, (0, current_position))  # (cost, position) 
        distances = {current_position: 0}
        previous_nodes = {current_position: None}

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # convert to coords
            target_x, target_y = self.Adress2Coords(target)

            if current_node == (target_x, target_y):
                break

            x, y = current_node
            # Check neighboring cells in 4 directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring cells
                nx, ny = x + dx, y + dy
                # Ensure within grid bounds and cell is safe
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and self.safe_grid[nx][ny] == 0:
                    new_cost = current_distance + 1
                    if (nx, ny) not in distances or new_cost < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_cost
                        previous_nodes[(nx, ny)] = (x, y)
                        heapq.heappush(pq, (new_cost, (nx, ny)))

         # Reconstruct the path from target to current position
        path = []
        step = (target_x, target_y)
        while step:
            path.append(self.coords2address(step))
            step = previous_nodes.get(step)
        path.reverse()

        return path
