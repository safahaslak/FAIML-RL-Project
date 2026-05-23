"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
"""Sample script for training a control policy on the Hopper environment"""
import gymnasium as gym
import numpy as np
import torch

# Import our custom agent and policy from agent.py
from agent import Policy, Agent 

def main():
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    # 1. Initialize the Agent
    state_dim = env.observation_space.shape[0]  # 11 for Hopper
    action_dim = env.action_space.shape[0]      # 3 for Hopper
    
    policy = Policy(state_dim, action_dim)
    agent = Agent(policy)

    n_episodes = 2000  # Number of episodes for training

    print("\n--- Starting REINFORCE Training ---\n")

    # 2. Main Training Loop
    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # The agent decides on an action based on the current state
            action, action_log_prob = agent.get_action(state)
            
            # Convert PyTorch Tensor to Numpy array for the Gymnasium environment
            action_np = action.detach().cpu().numpy()
            
            # Take a step in the environment
            next_state, reward, terminated, truncated, info = env.step(action_np)
            
            # In Gymnasium v26+, an episode ends if terminated or truncated
            done = terminated or truncated
            
            # Store the outcome in the agent's memory for policy update
            agent.store_outcome(state, next_state, action_log_prob, reward, done)
            
            state = next_state
            episode_reward += reward

        # Episode is over. Update the policy using the collected trajectory!
        agent.update_policy()

        # Print training progress every 20 episodes
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1:4d} | Total Reward: {episode_reward:8.2f}")

        if (ep + 1) == n_episodes:
            torch.save(agent.policy.state_dict(), "hopper_policy_ep1000.pth")
            print("\n(hopper_policy_ep1000.pth) Model Saved Successfully!\n")

if __name__ == '__main__':
    main()