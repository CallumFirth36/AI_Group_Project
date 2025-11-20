import numpy as np
from GroupProjectRL import ProjectEnv

def run(episodes, is_training=True, render=False):
    # Example grid
    grid = np.array([
        ['R','0','0','0','G'],
        ['1','1','0','0','1'],
        ['0','1','1','1','1'],
        ['0','0','1','0','1'],
        ['Y','1','1','0','B']
    ])

    env = ProjectEnv(grid, render_mode="human" if render else None)

    q = np.zeros((env.size * env.size, env.action_space.n))

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 0.5
    epsilon_decay = 0.001
    rng = np.random.default_rng()

    for i in range(episodes):
        obs, info = env.reset()
        state = obs["agent"][0] * env.size + obs["agent"][1]
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_state = new_obs["agent"][0] * env.size + new_obs["agent"][1]

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
    # example grid
    grid = np.array([
        ['R','0','0','0','G'],
        ['1','1','0','0','1'],
        ['0','1','1','1','1'],
        ['0','0','1','0','1'],
        ['Y','1','1','0','B']
    ])

    env = ProjectEnv(grid, render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        state = obs["agent"][0] * env.size + obs["agent"][1]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = np.argmax(q[state, :])  # greedy choice
            new_obs, reward, terminated, truncated, _ = env.step(action)
            state = new_obs["agent"][0] * env.size + new_obs["agent"][1]

        print(f"Playback episode {ep+1} finished")

    env.close()

if __name__ == "__main__":
    q_table = run(episodes=500, is_training=True, render=True)
    playback(q_table, episodes=1)
