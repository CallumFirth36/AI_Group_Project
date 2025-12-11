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
    """Run RL model and display path - optimized version"""
    print("\nRunning Reinforcement Learning model...")
    print("Training Q-table (this may take a moment)...")
    
    # Create a temporary environment for training (no rendering for speed)
    train_env = MapTraversalEnvironment(renderMode=None)
    
    # Import the optimized training function
    from NewRLModel import run as trainRL, findPathFromQTable
    
    # Train Q-table (optimized, no rendering)
    print("Training in progress... (no rendering for speed)")
    # Reduced episodes for faster execution - can be increased for better paths
    q_table = trainRL(episodes=250, is_training=True, render=False, env=train_env)
    train_env.close()
    
    print("Finding optimal path using trained Q-table...")
    
    # Create a pathfinding environment (no rendering during pathfinding)
    path_env = MapTraversalEnvironment(renderMode=None)
    path_env.setStart(start_pos)
    path_env.setTarget(end_pos)
    
    # Find path without rendering (like A*)
    path = findPathFromQTable(q_table, path_env, start_pos, end_pos)
    path_env.close()
    
    # Reset main environment map to show clean path
    env.resetMap()
    env.setStart(start_pos)
    env.setTarget(end_pos)
    
    if path and len(path) > 1:
        print(f"Fastest path found! Length: {len(path)} steps")
        # Display path all at once (like A*) - path will be drawn in RED
        env.drawPath(path)
        env.render()
        print("Fastest route displayed in RED. Press any key to continue...")
        
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

def main():
    # Initialize environment
    env = MapTraversalEnvironment(renderMode="human")
    
    # Don't reset - just show the map (positions will be set by user)
    # Initialize map image
    if env.window is not None:
        env.mapImage = env.mapSurface.convert()
    else:
        env.mapImage = env.mapSurface
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
