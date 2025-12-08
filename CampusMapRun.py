import sys 
sys.path.append('../')
from CampusTraversalEnvironment import MapTraversalEnvironment
from randomSearchBot import RandomSearchBot

you = RandomSearchBot()
env = MapTraversalEnvironment(renderMode="human")

obs = env.getObs(False)
while not obs[3]:
    action = you.move(obs)
    obs = env.step(action)
    env.render()
    if(obs[3]):
        obs = env.reset()
