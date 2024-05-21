import pygame
import sys
import threading
import heapq
from pygame.locals import *
import random
from threading import Lock
from enum import Enum

pygame.init()

# City parameters
GRID_WIDTH, GRID_HEIGHT = 4, 4
BOX_WIDTH, BOX_HEIGHT = 1, 1
INITIAL_X,  INITIAL_Y = 0, 0 
INITIAL_SAFE = True

# UI parmeters
WINDOW_WIDTH, WINDOW_HEIGHT = 2048, 2048
TILE_SIZE = WINDOW_WIDTH // GRID_WIDTH
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('City Grid')
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
font = pygame.font.Font(None, 24)

try:
    tile_image = pygame.image.load('tile_city.jpg').convert()
    tile_image = pygame.transform.scale(tile_image, (TILE_SIZE, TILE_SIZE))
    drone_image = pygame.image.load('spider_drone.png').convert_alpha()  
    drone_image = pygame.transform.scale(drone_image, (120, 120))  # Scale image to 120x120 pixels
    bot_image = pygame.image.load('spider_bot.png').convert_alpha()
    bot_image = pygame.transform.scale(bot_image, (200, 150))  # Scale image to 200x200 pixels
    both_images = pygame.image.load('both_spiders.png').convert_alpha()
    both_images = pygame.transform.scale(both_images, (200, 200))  # Scale image to 200x200 pixels
except Exception as e:
    print(f"Failed to load image: {e}")
    sys.exit()

class Status(Enum):
    SAFE = 1
    UNSAFE = 2
    GOAL = 3

class Intersection:
    def __init__(self, x, y, status=Status.SAFE):
        self.x = x
        self.y = y
        self.odom = None
        self.neighbors = []
        self.status = status

    def add_neighbor(self, neighbor):
        # Check if neighbor is already in the list
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def __str__(self):
        return f"Intersection at ({self.x}, {self.y}) with status {self.status}, Neighbors: {[f'({n.x}, {n.y})' for n in self.neighbors]}"

class CityGrid:
    def __init__(self, size_x, size_y, square_size_x, square_size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.square_size_x = square_size_x
        self.square_size_y = square_size_y
        self.grid = [[None for _ in range(size_y)] for _ in range(size_x)]

    def add_intersection(self, x, y, status):
        if self.grid[x][y] is None:
            self.grid[x][y] = Intersection(x, y, status)
        else:
            # Only update the status if it's different and valid
            if isinstance(status, Status) and self.grid[x][y].status != status:
                self.grid[x][y].status = status
        self.connect_intersections(x, y)
        return self.grid[x][y]  # Return the intersection object

    def get_intersection(self, x, y):
        return self.grid[x][y]
    
    def generate_random_map(self, num_unsafe, goal_position):
        # Initialize all intersections
        for x in range(self.size_x):
            for y in range(self.size_y):
                status = Status.SAFE
                self.add_intersection(x, y, status)

        # Set specific statuses
        goal_x, goal_y = goal_position
        self.grid[goal_x][goal_y].status = Status.GOAL
        self.grid[INITIAL_X][INITIAL_Y].status = Status.SAFE

        unsafe_positions = random.sample(
            [(x, y) for x in range(self.size_x) for y in range(self.size_y)
             if (x, y) != goal_position and (x, y) != (INITIAL_X, INITIAL_Y)],
            num_unsafe
        )
        for x, y in unsafe_positions:
            self.grid[x][y].status = Status.UNSAFE

        # Connect the initial and goal positions explicitly
        self.connect_intersections(INITIAL_X, INITIAL_Y)
        self.connect_intersections(goal_x, goal_y)

    def connect_intersections(self, x, y):
        current = self.grid[x][y]

        # Check and add neighbors
        if x > 0 and self.grid[x - 1][y]:
            current.add_neighbor(self.grid[x - 1][y])
            self.grid[x - 1][y].add_neighbor(current)
        if x < self.size_x - 1 and self.grid[x + 1][y]:
            current.add_neighbor(self.grid[x + 1][y])
            self.grid[x + 1][y].add_neighbor(current)
        if y > 0 and self.grid[x][y - 1]:
            current.add_neighbor(self.grid[x][y - 1])
            self.grid[x][y - 1].add_neighbor(current)
        if y < self.size_y - 1 and self.grid[x][y + 1]:
            current.add_neighbor(self.grid[x][y + 1])
            self.grid[x][y + 1].add_neighbor(current)

    def display_city_neighbors(self):
        for row in self.grid:
            for intersection in row:
                if intersection and intersection.odom:
                    print(intersection)

    def print_map(self):
        # This method will print the current status map of the grid
        map_str = ""
        for y in range(self.size_y):
            row_str = ""
            for x in range(self.size_x):
                if self.grid[x][y] is None:
                    row_str += " None "
                else:
                    # Check if the status is indeed an instance of Status
                    if isinstance(self.grid[x][y].status, Status):
                        row_str += f" {self.grid[x][y].status.name[:4]} "
                    else:
                        row_str += " ??? "  # Placeholder when status is not a Status instance
            map_str += row_str + "\n"
        print(map_str)



class SpiderDrone:
    def __init__(self, city_grid):
        self.city_grid = city_grid
        self.current_intersection = self.city_grid.add_intersection(INITIAL_X, INITIAL_Y, Status.SAFE)
        self.drone_image = drone_image
        self.both_image = both_images
        self.map_defined = self.check_map_defined()
        self.lock = Lock()

    def define_map(self, num_unsafe):
        self.city_grid.initialize_unsafe_intersections(num_unsafe)
        self.map_defined = True
        print("Map has been fully defined with unsafe intersections.")

    def move_to_intersection(self, x, y, status):
        with self.lock:
            if not isinstance(status, Status):
                print(f"Error: Invalid status {status}")
                return
            
            # Check if the grid coordinates are within the city grid
            if not (0 <= x < self.city_grid.size_x) or not (0 <= y < self.city_grid.size_y):
                print("Invalid coordinate")
                return

            # Check if the move is to a neighboring intersection or the same one (for updating purposes)
            if self.current_intersection is not None:
                current_x, current_y = self.current_intersection.x, self.current_intersection.y
                # Allow updating the same intersection or ensure the move is to a direct neighbor
                if not (abs(current_x - x) + abs(current_y - y) == 1 or (current_x == x and current_y == y)):
                    print("Invalid move: You can only move to a direct neighboring intersection or update the current one.")
                    return
            
            # Initialize or update the intersection's status
            intersection_exists = self.city_grid.grid[x][y] is not None
            if not intersection_exists:
                self.city_grid.add_intersection(x, y, status)
            else:
                # Update the status if it's different from the existing one
                if self.city_grid.grid[x][y].status != status:
                    self.city_grid.grid[x][y].status = status

            # Update the current intersection and its properties
            self.current_intersection = self.city_grid.grid[x][y]
            self.current_intersection.odom = True

            # Connect neighbors every time in case of new intersection or status changes
            self.city_grid.connect_intersections(x, y)

            # Check if the map is fully defined after this move
            self.check_map_defined()


    def check_map_defined(self):
        # Check if all intersections have been defined (either safe or unsafe)
        all_defined = all(intersection is not None for row in self.city_grid.grid for intersection in row)
        if all_defined:
            self.map_defined = True
            print("Map is now fully defined.")
            return True
        return False

    def get_current_position(self):
        if self.current_intersection:
            return self.current_intersection.x, self.current_intersection.y
        else:
            return None, None
        
    def draw(self, window):
        if self.current_intersection:
            drone_pos = (self.current_intersection.x * TILE_SIZE + (TILE_SIZE - 120) // 2,
                         self.current_intersection.y * TILE_SIZE + (TILE_SIZE - 120) // 2)
            window.blit(self.drone_image, drone_pos)

    def draw_both(self, window):
        if self.current_intersection:
            drone_pos = (self.current_intersection.x * TILE_SIZE + (TILE_SIZE - 120) // 2,
                         self.current_intersection.y * TILE_SIZE + (TILE_SIZE - 120) // 2)
            window.blit(self.both_image, drone_pos)
    

def handle_input(spider_drone):
    while True:
        x = int(input("Enter x coordinate (0-4): "))
        y = int(input("Enter y coordinate (0-4): "))
        
        status_input = input("Enter status of the intersection (Safe/Unsafe/Goal): ").strip().lower()
        
        # Map user input to Status enum correctly
        status_map = {
            "safe": Status.SAFE,
            "unsafe": Status.UNSAFE,
            "goal": Status.GOAL
        }
        status = status_map.get(status_input)
        
        if status is None:
            print("Invalid status. Please enter 'Safe', 'Unsafe', or 'Goal'.")
            continue  # Skip the rest of the loop to ask for input again
        
        if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
            spider_drone.move_to_intersection(x, y, status)
            if spider_drone.current_intersection:
                city.display_city_neighbors()
                city.print_map()
            else:
                print("Current intersection is not defined yet.")
        else:
            print("Coordinates out of grid bounds, please enter valid coordinates within the grid size.")



class SpiderBot:
    def __init__(self, city_grid, initial_position):
        self.city_grid = city_grid
        self.current_intersection = self.city_grid.add_intersection(INITIAL_X, INITIAL_Y, Status.SAFE)
        print(f"Initial status at ({INITIAL_X}, {INITIAL_Y}): {self.current_intersection.status}")
        self.bot_image = bot_image
        if initial_position:
            self.set_position(*initial_position)

    def set_position(self, x, y):
        """ Set the initial position of the SpiderBot. """
        if 0 <= x < self.city_grid.size_x and 0 <= y < self.city_grid.size_y:
            self.current_position = (x, y)
        else:
            print("Invalid or unsafe starting position.")

    def find_shortest_path(self, target_x, target_y):
        """ Finds the shortest path from current position to the target using Dijkstra's algorithm. """
        if not self.current_position:
            print("Initial position not set.")
            return []

        if not (0 <= target_x < self.city_grid.size_x) and (0 <= target_y < self.city_grid.size_y):
            print("Target position out of bounds.")
            return []

        # Using a priority queue to implement Dijkstra's algorithm
        pq = []
        heapq.heappush(pq, (0, self.current_position))  # (cost, position)
        distances = {self.current_position: 0}
        previous_nodes = {self.current_position: None}

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_node == (target_x, target_y):
                break

            x, y = current_node
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring cells
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.city_grid.size_x and 0 <= ny < self.city_grid.size_y and self.city_grid.grid[nx][ny].safe:
                    new_cost = current_distance + 1
                    if (nx, ny) not in distances or new_cost < distances[(nx, ny)]:
                        distances[(nx, ny)] = new_cost
                        previous_nodes[(nx, ny)] = (x, y)
                        heapq.heappush(pq, (new_cost, (nx, ny)))

        # Reconstruct the path
        path = []
        step = (target_x, target_y)
        while step:
            path.append(step)
            step = previous_nodes.get(step)
        path.reverse()

        return path

    def move_along_path(self, path):
        """ Simulate movement along the path. """
        for position in path:
            self.current_position = position
            print(f"Moved to {position}")

    def draw(self, window):
        if self.current_intersection:
            drone_pos = (self.current_intersection.x * TILE_SIZE + (TILE_SIZE - 200) // 2,
                         self.current_intersection.y * TILE_SIZE + (TILE_SIZE - 200) // 2)
            window.blit(self.bot_image, drone_pos)

    def get_current_position(self):
        if self.current_intersection:
            return self.current_intersection.x, self.current_intersection.y
        else:
            return None, None

def main():
    global city
    city = CityGrid(GRID_WIDTH, GRID_HEIGHT, BOX_WIDTH, BOX_HEIGHT)
    city.add_intersection(INITIAL_X, INITIAL_Y, Status.SAFE)
    city.generate_random_map(3, (3,3))  # debugging function will randomly generate a map
    spider_drone = SpiderDrone(city)
    
    # Initialize spider_bot here
    spider_bot = SpiderBot(city, (INITIAL_X, INITIAL_Y))

    # Check initial values are correct
    if spider_drone.map_defined:
        print("Map is fully defined at startup.")
    elif 0 <= INITIAL_X < GRID_WIDTH and 0 <= INITIAL_Y < GRID_HEIGHT:
        spider_drone.move_to_intersection(INITIAL_X, INITIAL_Y, INITIAL_SAFE)
    else:
        print("Invalid initial position. Please restart and choose coordinates within the grid.")
        sys.exit()

    # Start input thread only if map is not predefined
    if not spider_drone.map_defined:
        input_thread = threading.Thread(target=handle_input, args=(spider_drone,))
        input_thread.start()

    running = True
    waiting_for_destination = False
    map_displayed = False
    destination_reached = False
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            
            if spider_drone.map_defined:
                waiting_for_destination = True

        if waiting_for_destination and map_displayed and not destination_reached:
            destination_x = int(input("Enter destination x coordinate (0-4): "))
            destination_y = int(input("Enter destination y coordinate (0-4): "))
            if 0 <= destination_x < GRID_WIDTH and 0 <= destination_y < GRID_HEIGHT:
                path = spider_bot.find_shortest_path(destination_x, destination_y)
                spider_bot.move_along_path(path)
                destination_reached = True
            else:
                print("Destination coordinates are out of bounds.")
            waiting_for_destination = False  # Reset the flag

        if destination_reached:
            print("Mission Success!")
            break

        window.fill(WHITE)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                window.blit(tile_image, (x * TILE_SIZE, y * TILE_SIZE))
                if city.grid[x][y]:
                    # Create a separate surface for the circle with an alpha channel
                    circle_surface = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    color = GREEN if city.grid[x][y].status == Status.SAFE else RED if city.grid[x][y].status == Status.UNSAFE else BLUE
                    # Set alpha value for transparency: 0 (fully transparent) to 255 (opaque)
                    alpha_value = 128  # 50% transparency
                    transparent_color = color + (alpha_value,)
                    # Draw the circle on the transparent surface
                    radius = TILE_SIZE // 3  # Increase the radius size to 1/3 of the tile size for visibility
                    pygame.draw.circle(circle_surface, transparent_color, (TILE_SIZE // 2, TILE_SIZE // 2), radius)
                    # Blit the transparent surface onto the main window
                    window.blit(circle_surface, (x * TILE_SIZE, y * TILE_SIZE))

        if spider_drone.get_current_position() == spider_bot.get_current_position():
            spider_drone.draw_both(window)  # You might need to implement this method
        else:
            # Draw each entity separately
            spider_drone.draw(window)
            spider_bot.draw(window)

        pygame.display.update()
        map_displayed = True


    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
