## The implementation of the RL machine enviroment learning from gymnasium

## Contributers
# Callum Firth 2635930
# Jonny Forbes 2643497
# Bailey Clark 2636229
# firstName lastName MatriculationNumber
## The implementation of the RL machine environment learning from gymnasium

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np 
from gymnasium.envs.registration import register
import pygame

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class MapTraversalEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, renderMode=None):
        pygame.init()
        
        self.width = 1280
        self.height = 720
        self.size = (1280, 720)
        self.renderMode = renderMode
        self.window = None

        if self.renderMode == "human":
            self.window = pygame.display.set_mode(self.size)

        self.filename = "campusMapAllRoadGreenLine.png"
            
        self.mapSurface = pygame.image.load(self.filename)
        
        if self.window is not None:
            self.mapImage = self.mapSurface.convert()
        else:
            self.mapImage = self.mapSurface

        self.map = np.zeros((self.width, self.height))
        self.generateGridValues()
        
        self.currentLocation = np.array([0, 0])
        self.targetLocation = np.array([0, 0])
        
        # Observation Space
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, max(self.width, self.height), shape=(2,), dtype=int),
            "target": spaces.Box(0, max(self.width, self.height), shape=(2,), dtype=int),
        })

        self.action_space = spaces.Discrete(5)

        # Action Space
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, -1]), 
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, 1]),
            4: np.array([0, 0])
        }

    # Get Observations from environment
    def getObs(self):
        return {"agent": self.currentLocation, "target": self.targetLocation}
    
    # Get Distance to goal
    def getInfo(self):
        return {
            "distance": np.linalg.norm(
                self.currentLocation - self.targetLocation, ord=1
            )
        }

    # Reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Check if options dict contains use_random flag
        use_random = True
        if options is not None and isinstance(options, dict):
            use_random = options.get('use_random', True)
        
        # Only reset positions if use_random is True
        # Otherwise, preserve existing positions
        if use_random:
            # Reset the output pygame map
            if self.window is not None:
                 self.mapImage = self.mapSurface.convert()
            else:
                 self.mapImage = self.mapSurface

            self.currentLocation = np.array([0, 0])
            self.targetLocation = np.array([0, 0])
            self.randomStartAndTarget()
        # If use_random is False, don't change positions or map

        return self.getObs(), self.getInfo()
    
    # Reset without randomizing positions (keeps current start/end)
    def resetMap(self):
        """Reset the map image without changing positions"""
        # Store current positions
        saved_start = self.currentLocation.copy() if np.any(self.currentLocation != [0, 0]) else None
        saved_target = self.targetLocation.copy() if np.any(self.targetLocation != [0, 0]) else None
        
        # Reset the map image
        if self.window is not None:
            self.mapImage = self.mapSurface.convert()
        else:
            self.mapImage = self.mapSurface
        
        # Restore and redraw markers
        if saved_start is not None:
            self.currentLocation = saved_start
            self.drawStartMarker(self.currentLocation)
        if saved_target is not None:
            self.targetLocation = saved_target
            self.drawEndMarker(self.targetLocation)

    # Set start 
    def setStart(self, startPos):
        self.currentLocation = np.array(startPos)
        self.drawStartMarker(self.currentLocation)

    # Set goal and draw marker on map
    def setTarget(self, targetPos):
        self.targetLocation = np.array(targetPos)
        self.drawEndMarker(self.targetLocation)
    
    # Draw blue marker (end) on map
    def drawEndMarker(self, location):
        x, y = location
        radius = 5
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i*i + j*j <= radius*radius:
                    px, py = x + i, y + j
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.mapImage.set_at((px, py), (0, 0, 255))  # Blue
    
    # Draw red marker (start) on map
    def drawStartMarker(self, location):
        x, y = location
        radius = 5
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i*i + j*j <= radius*radius:
                    px, py = x + i, y + j
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.mapImage.set_at((px, py), (255, 0, 0))  # Red
    
    # Draw marker on map (legacy method for compatibility)
    def drawMarker(self, location):
        self.drawEndMarker(location)
    
    # Find nearest valid pixel to a click position
    # clickPos is in pygame format (x, y) which corresponds to (col, row)
    def findNearestValidPixel(self, clickPos, searchRadius=50):
        x, y = clickPos  # x is column, y is row
        best_pos = None
        best_dist = float('inf')
        
        # Convert to map coordinates: map uses [row][col] = [y][x]
        for row in range(max(0, y - searchRadius), min(self.height, y + searchRadius)):
            for col in range(max(0, x - searchRadius), min(self.width, x + searchRadius)):
                if self.map[col][row] == 1:  # Note: map is [col][row] based on generateGridValues
                    dist = np.sqrt((col - x)**2 + (row - y)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (col, row)  # Return as (col, row) = (x, y) in map coordinates
        
        return best_pos
    
    # Get valid neighbors for pathfinding
    def getNeighbors(self, pos):
        x, y = pos
        neighbors = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Right, Left, Down, Up
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.validateMove((new_x, new_y)):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    # Draw complete path
    def drawPath(self, path):
        """Draw a complete path as a red line"""
        if path is None or len(path) == 0:
            return
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            self.drawLine(start, end, (255, 0, 0))
    
    # Draw a line between two points
    def drawLine(self, start, end, color):
        """Bresenham's line algorithm"""
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.mapImage.set_at((x, y), color)
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # Set random start and goal
    def randomStartAndTarget(self):
        valid_pixels = np.argwhere(self.map == 1)

        indices = np.random.choice(len(valid_pixels), 2, replace=False)
        startPos = valid_pixels[indices[0]]
        targetPos = valid_pixels[indices[1]]
        self.setStart(startPos)
        self.setTarget(targetPos)

    # Validate Move
    def validateMove(self, newLoc):
        if 0 <= newLoc[0] < self.width and 0 <= newLoc[1] < self.height:
            if self.map[newLoc[0]][newLoc[1]] == 1:
                return True
        return False

    def generateGridValues(self):
        for x in range(self.width):
            for y in range(self.height):
                pixelColour = self.mapSurface.get_at((x, y))
                if pixelColour == (0, 255, 0):
                    self.map[x][y] = 1

    def step(self, action, render_path=False):
        reward = -1 
        terminated = False
        truncated = False
        
        # Check Distance to goal when entering building and set reward
        if action == 4:
            # Don't really know how this works but stackoverflow told me it gets the difference between two coordinates using numpys linear algebra libs
            dist = np.linalg.norm(self.currentLocation - self.targetLocation)
            if dist < 10.0:
                reward = 100
                terminated = True
            else:
                reward = -10
        else:
            direction = self._action_to_direction[action]
            new_loc = self.currentLocation + direction
            
            if self.validateMove(new_loc):
                self.currentLocation = new_loc
                if render_path:
                    self.drawStepPath()
            else:
                reward = -2 

        observation = self.getObs()
        info = self.getInfo()
        
        if self.renderMode == "human" and render_path:
            self.renderFrame()
            
        return observation, reward, terminated, truncated, info
    
    # Render frame
    def renderFrame(self):
        if self.window:
            pygame.event.pump()
            self.window.blit(self.mapImage, (0, 0))
            pygame.display.flip()
    
    # Render method for gymnasium API compatibility
    def render(self):
        if self.renderMode == "human":
            self.renderFrame()
        elif self.renderMode == "rgb_array":
            # Return RGB array if needed in the future
            pass

    # Draw redLine for where model has gone (legacy method for step-by-step)
    def drawStepPath(self):
        x, y = self.currentLocation
        if 0 <= x < self.width and 0 <= y < self.height:
            self.mapImage.set_at((x, y), (255, 0, 0))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

register(
    id="GroupProject-v0",
    entry_point="MapTraversal:MapTraversalEnvironment",
    max_episode_steps=2000,
)

