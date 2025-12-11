import numpy as np
from MapTraversal import MapTraversalEnvironment

def run(episodes, is_training=True, render=False):

    env = MapTraversalEnvironment(renderMode="human")

    q = np.zeros((env.width * env.height, env.action_space.n)) 

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 0.5
    epsilon_decay = 0.001
    rng = np.random.default_rng()

    for i in range(episodes):
        obs, info = env.reset()
        
        state = obs["agent"][1] * env.width + obs["agent"][0] 
        terminated = False
        truncated = False
        total_reward = 0

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
            total_reward += reward

        epsilon = max(epsilon - epsilon_decay, 0)

        print(f"Episode {i+1}/{episodes}, total reward: {total_reward}")

    env.close()
    return q

def playback(q, episodes=5):

    env = MapTraversalEnvironment(renderMode="human")

    for ep in range(episodes):
        obs, info = env.reset()
 
        state = obs["agent"][1] * env.width + obs["agent"][0] 
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = np.argmax(q[state, :])  # greedy choice
            new_obs, reward, terminated, truncated, _ = env.step(action)

            state = new_obs["agent"][1] * env.width + new_obs["agent"][0] 

        print(f"Playback episode {ep+1} finished")

    env.close()

if __name__ == "__main__":
    q_table = run(episodes=500, is_training=True, render=True)
    playback(q_table, episodes=1)