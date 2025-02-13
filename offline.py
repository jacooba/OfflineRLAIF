import d3rlpy
import minari
import os
import imageio
import gym
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend on Mac
import matplotlib.pyplot as plt
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import EnvironmentEvaluator

STITCHED_PATH = "Pendulum_Stitched.h5"
EXPERT_PATH = "Pendulum_Expert.d3"

def train(seed, data_name="Pendulum_Stitched", algo="awac", vis_data=False, num_steps=500):
    assert data_name in ["Pendulum-v1", "Pendulum_Stitched", "antmaze-medium-play-v0"], "Invalid environment name"
    assert algo in ["awac", "bc", "td3+bc"], "Invalid agent name"

    set_seed(seed)

    # Load dataset and environment
    if data_name == "Pendulum-v1":
        from d3rlpy.datasets import get_pendulum
        mdp_dataset, env = get_pendulum()
        env_name = "Pendulum-v1"
    elif data_name == "Pendulum_Stitched":
        with open(STITCHED_PATH, "rb") as f:
            mdp_dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env_name = "Pendulum-v1"
    elif data_name == "antmaze-medium-play-v0":
        print("Currently Debugging this environment; may not work.")
        dataset = minari.load_dataset("D4RL/antmaze/medium-play-v1")
        env = dataset.recover_environment()
        def flatten_observation(obs):
            return np.concatenate([obs['achieved_goal'], obs['desired_goal'], obs['observation']])
        env = gym.wrappers.TransformObservation(env, flatten_observation)
        episodes = list(dataset.iterate_episodes())
        print("Number of episodes:", len(episodes))

        # Extract data
        # with properties 'id', 'observations', 'actions', 'rewards', 'terminations', 'truncations', 'infos']
        observations = np.vstack([ep.observations["observation"] for ep in episodes])
        actions = np.vstack([ep.actions for ep in episodes])
        rewards = np.hstack([ep.rewards for ep in episodes])
        terminals = np.hstack([ep.terminations for ep in episodes])
        truncations = np.hstack([ep.truncations for ep in episodes])
        print("Number of observations in episode 0:", len(episodes[0].observations["observation"]))
        print("Number of actions in episode 0:", len(episodes[0].actions))
        assert len(episodes[0].observations["observation"]) == len(episodes[0].actions) + 1

        # Convert to d3rlpy format
        mdp_dataset = MDPDataset(observations, actions, rewards, terminals, truncations)
        print("Dataset size:", len(mdp_dataset.episodes))
        # import pdb; pdb.set_trace()
        print("MDP Dataset Episodes:", len(mdp_dataset.episodes))
        assert len(mdp_dataset.episodes) > 0, "ERROR: No episodes found in MDPDataset!"
        print("Sample episode:", mdp_dataset.episodes[:5])  # Print first 5 episodes
        env_name = "antmaze-medium-play-v0"
    print("Dataset size:", len(mdp_dataset.episodes))

    # Visualize data
    if vis_data:
        visualize_data(env_name, mdp_dataset, data_name)

    # Initialize model
    if algo == "awac":
        agent = d3rlpy.algos.AWACConfig().create(device="mps")
    elif algo == "bc":
        agent = d3rlpy.algos.BCConfig().create(device="mps")
    elif algo == "td3+bc":
        agent = d3rlpy.algos.TD3PlusBCConfig().create(device="mps")
    print("model initialized:", agent)

    # Ensure the dataset is not empty
    if len(mdp_dataset.episodes) == 0:
        raise ValueError("The MDPDataset is empty. Please check the dataset loading process.")

    # For evaluation
    env_evaluator = EnvironmentEvaluator(env)

    # Seed environment for reproducibility
    env.reset(seed=seed)
    d3rlpy.envs.seed_env(env, seed)

    # Run one training step to check if it works
    agent.build_with_dataset(mdp_dataset)
    history = agent.fit(mdp_dataset, n_steps=num_steps, n_steps_per_epoch=50, show_progress=True, 
        evaluators={'environment': env_evaluator}, logger_adapter=d3rlpy.logging.NoopAdapterFactory())
    # Alternatively, Use `fitter()` for step-by-step training
    # for epoch, metrics in enumerate(awac.fitter(mdp_dataset, n_steps=5000, show_progress=True, evaluators={'environment': env_evaluator})):
    #     print(f"Epoch {epoch}: {metrics}")
    #     # Optionally, break early if training starts logging NaNs
    #     if np.isnan(metrics["loss"]):
    #         print("NaN detected, stopping training early.")
    #         break
    print("Done")

    return history, agent, env, env_name

def plot(history, plot_filename):

    epochs = list(range(len(history)))

    plt.figure(figsize=(12, 4))

    # Plot Critic Loss
    if "critic_loss" in history[0][1]:
        critic_losses = [info["critic_loss"] for step, info in history]
        plt.subplot(1, 3, 1)
        plt.plot(epochs, critic_losses, label="Critic Loss", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Critic Loss Over Training")
        plt.legend()

    # Plot Actor Loss
    actor_key = "actor_loss" if "actor_loss" in history[0][1] else "loss"
    actor_losses = [info[actor_key] for step, info in history]
    plt.subplot(1, 3, 2)
    plt.plot(actor_losses, label="Actor Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Actor Loss Over Training")
    plt.legend()

    # Plot Environment Rewards
    env_rewards = [info["environment"] for step, info in history]
    plt.subplot(1, 3, 3)
    plt.plot(epochs, env_rewards, label="Environment Reward", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Reward")
    plt.title("Reward Over Training")
    plt.legend()

    plt.tight_layout()
    
    # Save plot
    plt.savefig(plot_filename)

def rollout(seed, agent, env_name, filename, n_episodes=1, fps=30, max_steps=1000):
    print(f"Generating rollout video: {filename}")
    set_seed(seed) 

    # Get env from env_name:
    if env_name == "Pendulum-v1":
        env = gym.make(env_name, render_mode="rgb_array")
    elif env_name == "antmaze-medium-play-v0":
        env = minari.load_dataset("D4RL/antmaze/medium-play-v1").recover_environment()
        def flatten_observation(obs):
            return np.concatenate([obs['achieved_goal'], obs['desired_goal'], obs['observation']])
        env = gym.wrappers.TransformObservation(env, flatten_observation)

    # Set environment seed
    d3rlpy.envs.seed_env(env, seed)

    frames = []
    observation, _ = env.reset(seed=seed)
    for _ in range(n_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = agent.predict(np.array([observation]))[0]  # Get policy action
            observation, reward, done, _, _ = env.step(action)
            frame = env.render()
            if frame is None:
                print("Warning: Environment did not return frames. Skipping video generation.")
                return
            frames.append(frame)
            print(f"Step: {len(frames)} Reward: {reward} Done: {done}")
            if len(frames) >= max_steps:
                break

    # Save video
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved rollout video at {filename}")


def set_seed(seed):
    """Set random seed globally for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gym.utils.seeding.np_random(seed)
    d3rlpy.seed(seed)

def generate_stitched_dataset(seed, dataset_name, expert_agent, n_episodes):
    """Generate dataset where a random half of each episode is expert and the other half is anti-expert."""
    set_seed(seed)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # Load expert agent
    # expert_agent = d3rlpy.load_learnable(expert_path) # Was having issues loading to mps device
    # expert_agent.build_with_env(env)

    observations, actions, rewards, terminations, truncations = [], [], [], [], []

    obs, _ = env.reset(seed=seed)
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_observations, episode_actions, episode_rewards = [], [], []
        done = False
        step = 0
        max_steps = 600 

        # Randomly decide which half is expert and which is anti-expert
        expert_first = random.choice([True, False])

        while not done and step < max_steps:
            print(f"Episode: {episode} Step: {step}")
            if (expert_first and step < max_steps // 2) or (not expert_first and step >= max_steps // 2):
                # Use Expert for this half
                action = expert_agent.predict(np.array([obs]))[0]
            else:
                # Use Anti-Expert (inverted actions)
                action = -10*expert_agent.predict(np.array([obs]))[0]
                # action = expert_agent.predict(np.array([obs]))[0] For debugging. put back when done.

            next_obs, reward, done, _, _ = env.step(action)

            episode_observations.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)

            obs = next_obs
            step += 1

        # `terminations`: False since episodes do not naturally terminate
        episode_terminations = np.zeros(len(episode_rewards), dtype=bool)
        episode_truncations = np.zeros(len(episode_rewards), dtype=bool)
        episode_truncations[-1] = True  # Last step is truncated, not terminated

        observations.append(episode_observations)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        terminations.append(episode_terminations)
        truncations.append(episode_truncations)

    # Convert lists to NumPy arrays
    observations = np.vstack(observations)
    actions = np.vstack(actions)
    rewards = np.hstack(rewards)
    terminations = np.hstack(terminations)
    truncations = np.hstack(truncations)

    # Convert to d3rlpy dataset
    mdp_dataset = MDPDataset(observations, actions, rewards, terminations, truncations)

    # Save dataset
    mdp_dataset.dump(dataset_name)
    print(f"Stitched dataset saved at {dataset_name}")

    # Visualize data
    visualize_data("Pendulum-v1", mdp_dataset, "Pendulum_Stitched", episodes=[0,1,2]) # episodes=[0, 1, 2, 3, 250, 499])

def visualize_data(env_name, mdp_dataset, data_name, episodes=[0, 250, 499]):
    for ep_index in episodes:
        print("Number of observations in episode:", len(mdp_dataset.episodes[ep_index].observations))
        render_env = gym.make(env_name, render_mode="rgb_array")
        frames = []
        for state in mdp_dataset.episodes[ep_index].observations:
            render_env.reset()
            render_env.unwrapped.state = np.arctan2(state[1], state[0]), state[2]
            frames.append(render_env.render())
        imageio.mimsave(f"{data_name}_episode{ep_index}.mp4", frames, fps=30)

def move_d3rl_agent_to_device(agent, device):
    """Move d3rlpy agent's internal model to a specified device (CPU/MPS/GPU)."""
    if hasattr(agent, "impl") and agent.impl is not None:
        for attr_name in dir(agent.impl._modules):
            module = getattr(agent.impl._modules, attr_name)
            if isinstance(module, torch.nn.Module):
                module.to(device)
        agent.impl._device = torch.device(device)
    else:
        raise RuntimeError("Agent implementation not initialized. Did you call `build_with_dataset()`?")


if __name__ == "__main__":
    seed = 20
    data_name="Pendulum_Stitched" # "Pendulum-v1", "Pendulum_Stitched"
    algo="td3+bc"

    # If there is no "Pendulum_Stitched" dataset locally, create it
    if data_name == "Pendulum_Stitched" and not os.path.exists(STITCHED_PATH):
        stitch_seed = 20
        set_seed(stitch_seed)
        # if there is no expert trained on original Pendulum dataset, train it
        # if not os.path.exists(EXPERT_PATH):
        print("Training expert on original Pendulum dataset...")
        _, agent, _, _ = train(stitch_seed, data_name="Pendulum-v1", algo="awac", vis_data=False, num_steps=500) ############ bc?
        for r_num in range(3): # Rollout expert
            rollout(stitch_seed+r_num, agent, "Pendulum-v1", f"expert_pendulum_s{stitch_seed+r_num}.mp4")
        # Move agent to CPU for faster rollout
        # Manually move the model to CPU
        # move_d3rl_agent_to_device(agent, "cpu") 
        # agent.save(EXPERT_PATH) # Save expert model. Loading from disk somehow makes the agent tun much faster, but incorrectly.
        # Generate stitched dataset
        print("Generating stitched dataset...")
        generate_stitched_dataset(stitch_seed, STITCHED_PATH, agent, 500) # 1
        # generate_stitched_dataset(stitch_seed, STITCHED_PATH, EXPERT_PATH, 500) # 1
        # move agent back to mps device
        # move agent back to mps device
        # move_d3rl_agent_to_device(agent, "mps") 
        print("Done generating data; please run the script again.")
        exit()

    set_seed(seed)

    # Train
    history, agent, env, env_name = train(seed, data_name=data_name, algo=algo, vis_data=False)
    # Plot
    plot(history, f"{algo}_plot_{data_name}_s{seed}.png")
    # Rollout
    rollout(seed, agent, env_name, f"{algo}_trained_{data_name}_s{seed}.mp4")