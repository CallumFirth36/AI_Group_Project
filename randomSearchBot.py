import numpy as np

WIDTH = 1280
HEIGHT = 720
MAP_SIZE = (WIDTH, HEIGHT)
PATHVALUE = 1

class RandomSearchBot:
    
    def __init__(self):
        self.map = np.zeros(MAP_SIZE)
        self.currentLocation = [0,0]
        self.targetLocation = [0,0]

    def getObs(self, observation):
        self.map = observation[0]
        self.currentLocation = observation[1]
        self.targetLocation = observation[2]

    def move(self, observation):
        
        directions = [[1,0],[0,1],[-1,0],[0,-1]]
        

        return directions[np.random.randint(0, 3)]



