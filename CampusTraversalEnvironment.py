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
        
        # Reset the output pygame map
        if self.window is not None:
             self.mapImage = self.mapSurface.convert()
        else:
             self.mapImage = self.mapSurface

        self.currentLocation = np.array([0, 0])
        self.targetLocation = np.array([0, 0])
        
        # Set random start and goal
        self.randomStartAndTarget()

        return self.getObs(), self.getInfo()

    # Set start 
    def setStart(self, startPos):
        self.currentLocation = np.array(startPos)

    # Set goal and draw marker on map
    def setTarget(self, targetPos):
        self.targetLocation = np.array(targetPos)
        self.drawMarker(self.targetLocation)
    
    # Draw blue marker on map
    def drawMarker(self, location):
        x, y = location
        for i in range(-2, 3):
            for j in range(-2, 3):
                if -1 < x+i < self.width and -1 < y+j < self.height:
                    self.mapImage.set_at((x+i, y+j),(0, 0, 255))

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

    def step(self, action):
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
                self.drawPath()
            else:
                reward = -2 

        observation = self.getObs()
        info = self.getInfo()
        
        if self.renderMode == "human":
            self.renderFrame()
            
        return observation, reward, terminated, truncated, info
    
    # Render frame
    def renderFrame(self):
        if self.window:
            pygame.event.pump()
            self.window.blit(self.mapImage, (0, 0))
            pygame.display.flip()

    # Draw redLine for where model has gone
    def drawPath(self):
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
