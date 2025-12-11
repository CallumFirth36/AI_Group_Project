import numpy as np
from CampusTraversalEnvironment import MapTraversalEnvironment

def run(episodes, is_training=True, render=False, env=None):
    """Optimized RL training function"""
    if env is None:
        env = MapTraversalEnvironment(renderMode="human" if render else None)
        close_env = True
    else:
        close_env = False

    # Optimize Q-table size - only track valid states
    q = np.zeros((env.width * env.height, env.action_space.n))
    
    # Optimized hyperparameters for faster convergence
    learning_rate = 0.15  # Balanced learning rate
    discount_factor = 0.95
    epsilon = 1.0  # Start with full exploration
    epsilon_decay = 0.998  # Faster decay for quicker exploitation
    epsilon_min = 0.05
    rng = np.random.default_rng()

    for i in range(episodes):
        obs, info = env.reset(options={'use_random': True})
        
        state = obs["agent"][1] * env.width + obs["agent"][0]
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        max_steps_per_episode = 2000

        while not (terminated or truncated) and step_count < max_steps_per_episode:
            # Epsilon-greedy action selection
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            # Step without rendering during training
            new_obs, reward, terminated, truncated, info = env.step(action, render_path=render)
            new_state = new_obs["agent"][1] * env.width + new_obs["agent"][0]

            if is_training:
                # Q-learning update
                best_next_action = np.max(q[new_state, :])
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * best_next_action - q[state, action]
                )

            state = new_state
            total_reward += reward
            step_count += 1

        # Decay epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Progress updates (less frequent for speed)
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Training episode {i+1}/{episodes}, epsilon: {epsilon:.3f}, avg reward: {total_reward/max(step_count,1):.2f}")

    if close_env:
        env.close()
    return q

def findPathFromQTable(q_table, env, start_pos, end_pos, max_steps=5000):
    """
    Find the FASTEST/OPTIMAL path using Q-table without rendering or drawing steps.
    Uses BFS with Q-value guidance to find shortest path efficiently.
    Returns the path as a list of positions.
    """
    from collections import deque
    
    # Set positions
    env.setStart(start_pos)
    env.setTarget(end_pos)
    obs = env.getObs()
    
    start_state = obs["agent"][1] * env.width + obs["agent"][0]
    goal_state = end_pos[1] * env.width + end_pos[0]
    goal_pos = np.array(end_pos)
    
    # Early termination if start == end
    if start_state == goal_state:
        return [tuple(start_pos)]
    
    # Helper function to get next state without stepping
    def getNextState(state, action):
        """Get next state without actually stepping in environment"""
        x = state % env.width
        y = state // env.width
        direction = env._action_to_direction[action]
        new_x, new_y = x + direction[0], y + direction[1]
        if 0 <= new_x < env.width and 0 <= new_y < env.height:
            if env.map[new_x][new_y] == 1:
                new_state = new_y * env.width + new_x
                return new_state, (new_x, new_y)
        return None, None
    
    # BFS to find shortest path, using Q-values to prioritize actions
    queue = deque([(start_state, [tuple(start_pos)], 0)])
    visited = {start_state: 0}  # state -> path_length
    best_path = None
    best_length = float('inf')
    
    # Limit search to prevent excessive computation
    max_visited = 10000
    
    while queue and len(visited) < max_visited:
        current_state, path, path_length = queue.popleft()
        
        # Check if we reached the goal
        current_pos = path[-1]
        dist = np.linalg.norm(np.array(current_pos) - goal_pos)
        if dist < 10.0 or current_state == goal_state:
            if path_length < best_length:
                best_path = path
                best_length = path_length
            # Continue to find potentially shorter paths
            if len(visited) > 5000:
                break
        
        # Get all possible actions, sorted by Q-value (highest first)
        actions_with_q = [(action, q_table[current_state, action]) 
                          for action in range(env.action_space.n)]
        actions_with_q.sort(key=lambda x: x[1], reverse=True)  # Sort by Q-value
        
        # Explore actions in order of Q-value
        for action, q_value in actions_with_q:
            next_state, next_pos = getNextState(current_state, action)
            
            if next_state is None:
                continue
            
            new_length = path_length + 1
            
            # Only explore if we haven't visited or found a shorter path
            if next_state not in visited or visited[next_state] > new_length:
                visited[next_state] = new_length
                new_path = path + [next_pos]
                queue.append((next_state, new_path, new_length))
        
        # Limit path length
        if path_length > max_steps:
            break
    
    # Return best path found, or fallback to greedy
    if best_path:
        return best_path
    
    # Fallback: use greedy approach if BFS didn't find path
    print("Warning: Optimal path search failed, using greedy path...")
    return findGreedyPathFromQTable(q_table, env, start_pos, end_pos, max_steps)

def findGreedyPathFromQTable(q_table, env, start_pos, end_pos, max_steps=5000):
    """Fallback greedy pathfinding"""
    env.setStart(start_pos)
    env.setTarget(end_pos)
    obs = env.getObs()
    
    path = [tuple(start_pos)]
    state = obs["agent"][1] * env.width + obs["agent"][0]
    visited = set()
    visited.add(state)
    
    steps = 0
    goal_pos = np.array(end_pos)
    
    while steps < max_steps:
        action = np.argmax(q_table[state, :])
        new_obs, reward, terminated, truncated, info = env.step(action, render_path=False)
        new_state = new_obs["agent"][1] * env.width + new_obs["agent"][0]
        current_pos = tuple(new_obs["agent"])
        
        path.append(current_pos)
        
        if terminated:
            break
        
        dist = np.linalg.norm(np.array(current_pos) - goal_pos)
        if dist < 10.0:
            break
        
        if new_state in visited:
            break
        visited.add(new_state)
        
        state = new_state
        steps += 1
    
    return path

def playback(q, episodes=5):
    env = MapTraversalEnvironment(renderMode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        state = obs["agent"][1] * env.width + obs["agent"][0]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = np.argmax(q[state, :])  # greedy choice
            new_obs, reward, terminated, truncated, info = env.step(action, render_path=True)
            state = new_obs["agent"][1] * env.width + new_obs["agent"][0]

        print(f"Playback episode {ep+1} finished")

    env.close()

if __name__ == "__main__":
    q_table = run(episodes=500, is_training=True, render=True)
    playback(q_table, episodes=1)