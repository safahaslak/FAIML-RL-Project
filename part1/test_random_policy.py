"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gymnasium as gym
import torch
import time
from agent import Policy, Agent

def main():
    render = True

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = Policy(state_dim, action_dim)
    agent = Agent(policy)
    agent.policy.load_state_dict(torch.load("hopper_policy_ep1000.pth"))
    print("Trained model (hopper_policy_ep1000.pth) loaded successfully")

    n_episodes = 50

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            # action = env.action_space.sample()  # Sample random action
            action_tensor, _ = agent.get_action(state, evaluation=True)
            action = action_tensor.detach().cpu().numpy()

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated

            if render:
                env.render()
                time.sleep(0.015)


if __name__ == '__main__':
    main()