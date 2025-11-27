## The implementation of the RL machine enviroment learning from gymnasium

## Contributers
# Callum Firth 2635930
# Jonny Forbes 2643497
# 
# firstName lastName MatriculationNumber


from enum import Enum
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np # Library for numerical computing
from gymnasium.envs.registration import register
import pygame

WIDTH = 1280
HEIGHT = 720
IMAGE_SIZE = (WIDTH, HEIGHT)
ROADCOLOURS = (200, 214, 226)
GRASSPATHCOLOURS = (166, 233, 194)


# movement directions
class Actions(Enum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3
    NORTHEAST = 4
    NORTHWEST = 5
    SOUTHEAST = 6
    SOUTHWEST = 7

# custom enviroment
class MapTraversalEnvironment(gym.Env):
    metadata = {"renderModes": ["human", "rgb_array"], "renderFPS": 50}

    #initialize enviroment
    def __init__(self, renderMode=None):
        # Image Size contains resolution of image "campusMapNoEntrances.png"
        self.size = IMAGE_SIZE
        self.window = pygame.display.set_mode(self.size)
        self.mapImage = pygame.image.load("campusMapAllRoadGreenLine.png").convert()
        self.map = self.generateGridValues()

        # observation space - just the grid, coordinates of current/target location and value of current/target location (Change value of high to match total different types of values in grid)- can make copy of grid inside of agent and do calculations there
        self.observation_space = spaces.Dict({
            "mapGrid": spaces.Box(low=0, high=self.size, shape=(self.size), dtype = np.int32),
            "currentLocation": spaces.Box(low=(0,0), high=self.size, shape=(2, ), dtype = np.int32),
            "targetLocation": spaces.Box(low=(0,0), high=self.size, shape=(2, ), dtype = np.int32),
        })

        # 9 actions: "north", "south", "east", "west", "northwest", "northeast", "southwest", "southeast", "enter building"
        self.action_space = spaces.Discrete(9)

        self.actionToDirection = {
            Actions.EAST.value: np.array([1, 0]),
            Actions.NORTH.value: np.array([0, 1]),
            Actions.WEST.value: np.array([-1, 0]),
            Actions.SOUTH.value: np.array([0, -1]),
            Actions.NORTHWEST.value: np.array([-1, 1]),
            Actions.NORTHEAST.value: np.array([1, 1]),
            Actions.SOUTHEAST.value: np.array([-1, -1]),
            Actions.SOUTHWEST.value: np.array([1, -1]),
            8: np.array([0, 0])
        }

        assert renderMode is None or renderMode in self.metadata["renderModes"]
        self.renderMode = renderMode
        self.clock = None

    # returns current observation of enviroment
    def getObs(self):
        pass

    # resets enviroment to initial state
    def reset(self, seed=None, options=None):
        self.generateGridValues()

    # return True or False
    def validateMove(self, action):
        pass

    # update current location and distance traversed
    def processMove(self, action):
        pass

    # set values of grid based on colours
    def generateGridValues(self):
        # Create a 2D list to hold the integer color values
        pixel_array_2d = []
        
        print(f"Image size: {WIDTH}x{HEIGHT}")
        print("Converting pixels to integers...")
        
        # Loop through every pixel
        for y in range(HEIGHT):
            row = []
            for x in range(WIDTH):
                # Get RGB value (ignores alpha if present)
                r, g, b, _ = self.mapImage.get_at((x, y))  # _ is alpha, we ignore it
                
                # Convert RGB to single 24-bit integer: 0xRRGGBB
                color_int = (r << 16) + (g << 8) + b
                
                row.append(color_int)
            pixel_array_2d.append(row)
        
        # Print the entire 2D array (warning: big images = lots of output!)
        print("\n2D Pixel Array (RGB as integers):")
        for row in pixel_array_2d:
            print(row)
        
        print("\nDone! Pixel data stored in 'pixel_array_2d'")

        return pixel_array_2d
    
    # check if current location = target and return True or False
    def hasArrived():
        if self.currentLocation == self.targetLocation
            return True
      return False

    # updates agent in enviroment
    def step(self, action):
        if self.validateMove(action):
            self.processMove(action)

            if self.hasArrived():
                self.renderEndScreen()
                return self.reward
    pass

    def waitForRender(self):
        time.sleep(1)

    # renders current state if enviroment
    def render(self):
        if self.render_mode == "rgb_array":
            return self.renderFrame()
    
    # render a frame
    # self.mapImage is a png converted into a pygame surface
    def renderFrame(self):
        if self.window is None and self.renderMode == "human":
            pygame.init()
            self.mapImage = pygame.image.load('campusMap.png').convert
            self.window.blit(self.mapImage, self.window.get_rect())
        if self.clock is None and self.renderMode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.update()
        self.waitForRender()
    
    def drawPath():
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# registers enviroment
register(
    id="GroupProject-v0", # unique name for your env
    entry_point="GroupProjectRL:ProjectEnv", # module:class path
)
