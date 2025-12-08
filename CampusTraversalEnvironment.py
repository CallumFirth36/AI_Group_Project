## The implementation of the RL machine enviroment learning from gymnasium

## Contributers
# Callum Firth 2635930
# Jonny Forbes 2643497
# Bailey Clark 2636229
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
ROADCOLOURS = (200, 214, 226), (197,212,225)
LOCATIONS = [] # Add Colours for building locations here (Probably something simple like (1,0,0) an increminting the first digit each time, I will add a key somewhere on the Git showing which building each value represents!
PATHVALUE = 1


# movement directions
class Actions(Enum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

# custom enviroment
class MapTraversalEnvironment(gym.Env):
    metadata = {"renderModes": ["human", "rgb_array"], "renderFPS": 50}

    #initialize enviroment
    def __init__(self, renderMode=None):
        # Image Size contains resolution of image "campusMapNoEntrances.png"
        self.size = IMAGE_SIZE
        self.window = pygame.display.set_mode(self.size)
        self.mapImage = pygame.image.load("campusMapAllRoadGreenLine.png").convert()
        self.map = np.zeros(self.size)
        self.generateGridValues()
        self.distanceTraversed = 0
        self.reward = 0
        self.currentLocation = [0,0]
        self.targetLocation = [0,0]
        self.randomStartAndTarget()

        assert renderMode is None or renderMode in self.metadata["renderModes"]
        self.renderMode = renderMode
        self.clock = None

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

    # Janky and needs change to actually use observation space but this works for now
        return self.map, self.currentLocation, self.targetLocation, hasArrived

    # resets enviroment to initial state
    def reset(self, seed=None, options=None):
        #Grid values do not need to be reset only the dispay image and new start/targets
        #we need to add the building locations to the Map as their own colours adn use them as the target locations not jsut random postions on green line, I will do that soon!
        super().reset(seed=seed)
        self.mapImage = pygame.image.load("campusMapAllRoadGreenLine.png").convert()
        self.randomStartAndTarget()

        return self.getObs(False)

    #Set starting location
    def setStart(self, startPos):
        self.currentLocation[0] = startPos[0]
        self.currentLocation[1] = startPos[1]
        return True

    
    #Set targetLocation
    def setTarget(self, targetPos):
        self.targetLocation[0] = targetPos[0]
        self.targetLocation[1] = targetPos[1]
        return True

    #Maybe used in future for allowing the user to click anywhere on the map for start and end? IDK might be difficult
    def setStartAndEndMouse(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                posx = pygame.mouse.get_pos()[0]    
                if not self.setStart(posx):
                    if self.setTarget(posx):
                        break

    # return True or False
    # Ensures search methods/Neural Networks cannot generate bad movements, Add negative reward here for training nueral network to make sure it never makes wrong moves.
    def validateMove(self, action):
        if self.map[self.currentLocation[0] + action[0]][self.currentLocation[1] + action[1]] == 1:
            return True
            
        return False
        
    # Very messy but sets a random location on the green line as the start and end point
    # Also updates the map and adds a blue square for where the end point is!
    def randomStartAndTarget(self):
        targetPos = [1,0]
        startPos = [1,0]

        while self.map[startPos[0]][startPos[1]] != 1:
            startPos = [np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)]
        while self.map[targetPos[0]][targetPos[1]] != 1:
            targetPos = [np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)]

        print(startPos,"startPos")
        print(targetPos,"targetPos")
        
        self.setStart(startPos)
        self.setTarget(targetPos)
        self.mapImage.set_at((self.targetLocation[0],self.targetLocation[1]) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0]+1,self.targetLocation[1]) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0]-1,self.targetLocation[1]) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0],self.targetLocation[1]+1) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0],self.targetLocation[1]-1) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0]+1,self.targetLocation[1]+1) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0]+1,self.targetLocation[1]-1) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0]-1,self.targetLocation[1]+1) , (0,0,255))
        self.mapImage.set_at((self.targetLocation[0]-1,self.targetLocation[1]-1) , (0,0,255))


    # update current location and distance traversed
    def processMove(self, action):
        # Updates current location and changes colour on map to red, actually looks nice!
        # I have included distance traversed counter incaser we want to display the distance on the end screen
        self.currentLocation[0] = self.currentLocation[0] + action[0]
        self.currentLocation[1] = self.currentLocation[1] + action[1]
        self.drawPath()
        self.distanceTraversed += 1

    # set values of grid based on colours
    def generateGridValues(self):
        # Simplified version of generate grid
        # Old one was good and worked but i think this one is easier to expand into adding the locations
        # once locations are added to the map i will add a loop that will check a list containing their colours and update the self.map grid with unique numbers
        for x in range(WIDTH):
            for y in range(HEIGHT):
                pixelColour = self.mapImage.get_at((x, y))
                if pixelColour == (0,255,0):
                    self.map[x][y] = PATHVALUE
    
    # check if current location = target and return True or False
    def hasArrived():
        if self.currentLocation == self.targetLocation
            return True
      return False

    # updates agent in enviroment
    # Probably needs changing but works for now
    def step(self, action):
        self.renderFrame()
        if(self.validateMove(action)):
            self.processMove(action)

        if(self.hasArrived()):
            self.renderEndScreen()
            return self.getObs(True)

        return self.getObs(False)

    # We should change wait for render depening on which agent is navigating the map, Could be a cool way to display the differences between them
    def waitForRender(self):
        time.sleep(0)

    # renders current state if enviroment
    def render(self):
        if self.render_mode == "rgb_array":
            return self.renderFrame()
    
    # render a frame
    # self.mapImage is a png converted into a pygame surface
    def renderFrame(self):
        if self.window is None and self.renderMode == "human":
            pygame.init()
            self.window.blit(self.mapImage, self.window.get_rect())
            pygame.display.update()
            self.waitForRender()
            
        if self.clock is None and self.renderMode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.update()
        self.waitForRender()
    
    def drawPath():
        for x in range(WIDTH):
            self.mapImage.set_at((self.currentLocation[0],self.currentLocation[1]) , (255,0,0))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# registers enviroment
register(
    id="GroupProject-v0", # unique name for your env
    entry_point="GroupProjectRL:ProjectEnv", # module:class path
)
