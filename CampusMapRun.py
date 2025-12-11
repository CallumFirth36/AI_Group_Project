import pygame
import numpy as np
from CampusTraversalEnvironment import MapTraversalEnvironment
from AStarPathfinding import AStarPathfinding

def waitForClick(env, message):
    """Wait for user to click on the map"""
    print(message)
    print("Click on the map to select location...")
    
    clock = pygame.time.Clock()
    waiting = True
    clicked_pos = None
    
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    clicked_pos = event.pos
                    waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
        
        env.render()
        clock.tick(60)
    
    return clicked_pos

def selectStartAndEnd(env):
    """Allow user to select start and end positions"""
    start_pos = None
    end_pos = None
    
    # Select start position
    while start_pos is None:
        click_pos = waitForClick(env, "Select START location (will be marked with RED dot)")
        if click_pos is None:
            return None, None
        
        # Find nearest valid pixel
        valid_pos = env.findNearestValidPixel(click_pos)
        if valid_pos:
            start_pos = valid_pos
            env.setStart(start_pos)
            env.render()
            print(f"Start position set to: {start_pos}")
        else:
            print("No valid path found near that location. Please try again.")
    
    # Select end position
    while end_pos is None:
        click_pos = waitForClick(env, "Select END location (will be marked with BLUE dot)")
        if click_pos is None:
            return start_pos, None
        
        # Find nearest valid pixel
        valid_pos = env.findNearestValidPixel(click_pos)
        if valid_pos:
            end_pos = valid_pos
            env.setTarget(end_pos)
            env.render()
            print(f"End position set to: {end_pos}")
        else:
            print("No valid path found near that location. Please try again.")
    
    return start_pos, end_pos

def chooseAlgorithm():
    """Prompt user to choose algorithm"""
    print("\n" + "="*50)
    print("Choose pathfinding algorithm:")
    print("1. Reinforcement Learning (Q-Learning)")
    print("2. A* Algorithm")
    print("="*50)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "RL"
        elif choice == "2":
            return "AStar"
        else:
            print("Invalid choice. Please enter 1 or 2.")

def runAStar(env, start_pos, end_pos):
    """Run A* algorithm and display path"""
    print("\nRunning A* pathfinding algorithm...")
    
    astar = AStarPathfinding(env)
    path = astar.find_path(start_pos, end_pos)
    
    if path:
        print(f"Path found! Length: {len(path)} steps")
        env.drawPath(path)
        env.render()
        print("Path displayed in red. Press any key to continue...")
        
        # Wait for user to see the path
        clock = pygame.time.Clock()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            clock.tick(60)
    else:
        print("No path found between start and end positions!")

def runRL(env, start_pos, end_pos):
    """Run RL model and display path"""
    print("\nRunning Reinforcement Learning model...")
    print("Training Q-table (this may take a moment)...")
    
    # Create a temporary environment for training
    train_env = MapTraversalEnvironment(renderMode=None)
    q_table = trainRLModel(train_env, episodes=200, is_training=True)
    train_env.close()
    
    print("Finding path using trained Q-table...")
    
    # Reset map and set custom positions
    env.resetMap()
    env.setStart(start_pos)
    env.setTarget(end_pos)
    obs = env.getObs()
    
    path = []
    state = obs["agent"][1] * env.width + obs["agent"][0]
    terminated = False
    truncated = False
    max_steps = 5000
    steps = 0
    visited = set()
    
    while not (terminated or truncated) and steps < max_steps:
        current_state = state
        if current_state in visited:
            # Avoid infinite loops
            break
        visited.add(current_state)
        
        action = np.argmax(q_table[state, :])
        new_obs, reward, terminated, truncated, info = env.step(action)
        
        current_pos = tuple(obs["agent"])
        path.append(current_pos)
        
        state = new_obs["agent"][1] * env.width + new_obs["agent"][0]
        obs = new_obs
        steps += 1
        
        # Check if we're close enough to goal
        dist = np.linalg.norm(np.array(new_obs["agent"]) - np.array(new_obs["target"]))
        if dist < 10.0:
            path.append(tuple(new_obs["agent"]))
            terminated = True
            break
    
    # Reset map to show clean path
    env.resetMap()
    env.setStart(start_pos)
    env.setTarget(end_pos)
    
    if path:
        print(f"Path found! Length: {len(path)} steps")
        env.drawPath(path)
        env.render()
        print("Path displayed in red. Press any key to continue...")
        
        # Wait for user to see the path
        clock = pygame.time.Clock()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            clock.tick(60)
    else:
        print("No path found!")

def trainRLModel(env, episodes=200, is_training=True):
    """Train RL model and return Q-table"""
    q = np.zeros((env.width * env.height, env.action_space.n))
    
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 0.5
    epsilon_decay = 0.001
    rng = np.random.default_rng()
    
    for i in range(episodes):
        obs, info = env.reset(options={'use_random': True})
        
        state = obs["agent"][1] * env.width + obs["agent"][0]
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
            
            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_state = new_obs["agent"][1] * env.width + new_obs["agent"][0]
            
            if is_training:
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )
            
            state = new_state
        
        epsilon = max(epsilon - epsilon_decay, 0)
        
        if (i + 1) % 50 == 0:
            print(f"Training episode {i+1}/{episodes}")
    
    return q

def main():
    # Initialize environment
    env = MapTraversalEnvironment(renderMode="human")
    
    # Reset to show the map without random positions
    env.reset(options={'use_random': False})
    env.render()
    
    print("="*50)
    print("Campus Map Pathfinding")
    print("="*50)
    
    # Select start and end positions
    start_pos, end_pos = selectStartAndEnd(env)
    
    if start_pos is None or end_pos is None:
        print("Start or end position not selected. Exiting...")
        env.close()
        return
    
    # Choose algorithm
    algorithm = chooseAlgorithm()
    
    # Run selected algorithm
    if algorithm == "AStar":
        runAStar(env, start_pos, end_pos)
    elif algorithm == "RL":
        runRL(env, start_pos, end_pos)
    
    print("\nProgram finished. Closing window...")
    pygame.time.wait(2000)  # Wait 2 seconds before closing
    env.close()

if __name__ == "__main__":
    main()
