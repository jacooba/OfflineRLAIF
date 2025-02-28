import d3rlpy
import minari
import os
import sys
import openai
import hashlib
import imageio
import base64
import gym
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend on Mac

import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from multiprocessing import Pool

from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import EnvironmentEvaluator

from weighted_bc import WBCConfig
from dpo import DPOConfig

STITCHED_PATH = "Pendulum_Stitched.h5"
OPENAI_API_KEY = None # Set Later
DEVICE = "mps"  # MPS imroves TD3+BC, minor slowdown for BC
SINGLE_SEED = False

def render_trajectory_base64(render_env, observations):
    base64_frames = []
    for state in observations:
        render_env.reset()
        render_env.unwrapped.state = np.arctan2(state[1], state[0]), state[2]
        frame = render_env.render()
        base64_image = render_frame_to_base64(frame)
        base64_frames.append(base64_image)
    return base64_frames, frame # Return base64_frames, last frame

def render_frame_to_base64(frame):
    """Convert a rendered frame (numpy array) to a base64-encoded string."""
    img = Image.fromarray(frame)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def train_and_eval(seed, lr, critic_learning_rate, save_rollout_videos, data_name, 
        algo, num_step, eval_during_training): 
    
    # Setup save dir
    save_dir = f"Results/{algo}"
    os.makedirs(save_dir, exist_ok=True)
    # Set seed
    set_seed(seed)

    # Train
    history, agent, _, env_name = train(seed, lr, critic_learning_rate,
                                        data_name=data_name, algo=algo, vis_data=False, num_steps=num_step,
                                        eval_during_training=eval_during_training,)
    # Plot
    plot(history, f"{save_dir}/plot_{data_name}_s{seed}.png")
    # Rollout
    r, succ = rollout(seed, agent, env_name, f"{save_dir}/trained_{data_name}_s{seed}.mp4", save_rollout_videos)

    return r, succ

class pref_agent:
    def __init__(self, learning_rate=1e-3, visualize_data=True, 
                 env_name="Pendulum-v1", subsample=20, subtrajectory_len=100,
                 sparse=False, loss_per_action=False): 
        self.subtrajectory_len = subtrajectory_len
        self.bc_agent = None
        self.subsample = subsample
        self.env_name = env_name
        self.visualize_data = visualize_data
        self.learning_rate = learning_rate
        self.sparse = sparse
        self.loss_per_action = loss_per_action
        if self.sparse:
            self.system_prompt = ("You are watching two videos of a red stick."
                                " The higher the stick, the better."
                                " Respond '1' if Video 1 is better, '2' if Video 2 is better.")
        else:
            self.system_prompt = ("You are watching two videos of a red stick."
                                " The goal is to swing the stick up to gain height."
                                " It is bad to let it fall and lose momentum."
                                " Respond '1' if Video 1 is better, '2' if Video 2 is better.")
        self.video1_prompt = "Video 1:"
        self.video2_prompt = "Video 2:"
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def fit(self, dataset, **kwargs):
        # Query VLM for preference scores, if not already done
        preferences = self.get_preferences(dataset)
        # Add preferences and other trajectories to batches in the dataset
        augmented_dataset = self.augment_dataset(dataset, preferences)
        # Make agent
        self.agent = DPOConfig(learning_rate=self.learning_rate, 
                               loss_per_action=self.loss_per_action,
                               obs_shape=(3,)).create(device=DEVICE)
        # Build the model with dataset
        self.agent.build_with_dataset(augmented_dataset)
        # Fit the model
        return self.agent.fit(augmented_dataset, **kwargs)

    def predict(self, observation):
        # Predict the action
        return self.agent.predict(observation)
    
    def augment_dataset(self, dataset, preferences):
        print(f"Creating augmented dataset with preferences")

        observations, actions, rewards, terminations, truncations = [], [], [], [], []

        num_episodes = len(dataset.episodes)
        for n, episode in enumerate(dataset.episodes):
            print(f"\nEpisode: {n+1}/{num_episodes}")
            episode_observations, episode_actions, episode_rewards = [], [], []
            episode_terminations, episode_truncations = [], []

            # Get VLM preferences score
            episode_preference_tuples = preferences[n]

            num_obs = len(episode.observations)
            assert num_obs % self.subtrajectory_len == 0, "Subtrajectory length must divide episode length"
            for i in range(0, len(episode.observations), self.subtrajectory_len):
                print(f"  Observation: {i}-{i+self.subtrajectory_len}/{num_obs}")
                sub_obs1 = episode.observations[i:i+self.subtrajectory_len]
                sub_act1 = episode.actions[i:i+self.subtrajectory_len]

                # Get VLM preferences
                index = i // self.subtrajectory_len
                pref, j, k = episode_preference_tuples[index]
                j, k = int(j), int(k)

                # Get the other subtrajectory
                sub_obs2 = dataset.episodes[j].observations[k:k+self.subtrajectory_len]
                sub_act2 = dataset.episodes[j].actions[k:k+self.subtrajectory_len]

                # Combine subtrajectories in feature dimension
                combined_obs = np.concatenate([sub_obs1, sub_obs2], axis=-1)
                combined_act = np.concatenate([sub_act1, sub_act2], axis=-1)
                # Add to dataset
                episode_observations.extend(combined_obs)
                episode_actions.extend(combined_act)
                episode_rewards.append(pref)
                episode_terminations.append(False)
                episode_truncations.append(False)

            observations.append(episode_observations)
            actions.append(episode_actions)
            rewards.append(episode_rewards)
            terminations.append(episode_terminations)
            episode_truncations[-1] = True  # Mark as truncated
            truncations.append(episode_truncations)

        # Convert lists to NumPy arrays
        # Flatten observations and actions from (episodes, num subtrajectories, subtrajectory_len, 2 trajectories * 3 obs dim)
        observations = np.stack(observations, axis=0)
        assert observations.shape == (num_episodes, num_obs, 2*3), "Invalid shape"
        actions = np.stack(actions, axis=0)
        assert actions.shape == (num_episodes, num_obs, 2), "Invalid shape"
        rewards = np.hstack(rewards)
        terminations = np.hstack(terminations)
        truncations = np.hstack(truncations)
    
        # Convert to d3rlpy dataset
        augmented_dataset = MDPDataset(observations, actions, rewards, terminations, truncations)

        return augmented_dataset
    
    def get_preferences(self, dataset):
        """
        Queries VLM for preference scores on dataset and returns the scores.
        """
        # Compute hash of dataset (using observations & actions)
        dataset_hash = hashlib.md5(np.stack(dataset.episodes[0].observations).tobytes()).hexdigest()
        algo_hash = hashlib.md5(f"{self.subtrajectory_len}_{self.subsample}".encode()).hexdigest()
        combined_hash = hashlib.md5(f"{dataset_hash}_{algo_hash}".encode()).hexdigest()
        identifier = combined_hash
        if self.sparse:
            identifier = "sparse_" + identifier
        if self.subtrajectory_len == 600:
            identifier = "full_" + identifier
        # Save path for numpty array of confidpreferenceence scores
        save_path = f"pref_{identifier}_vlm.npy"

        # Check if the preference scores already exist
        num_loaded = 0
        if os.path.exists(save_path):
            # If the scores are complete, load them instead of re-querying
            loaded_scores = np.load(save_path)
            num_loaded = len(loaded_scores)
            print(f"Found {num_loaded} VLM preference scores saved at {save_path}")
            if num_loaded == len(dataset.episodes):
                print(f"VLM preference scores are complete; loading instead.")
                return loaded_scores
            print(f"VLM preference scores are incomplete; resuming queries...")
            preference_tuples = list(loaded_scores)
        else:
            print(f"Querying VLM for preference scores and saving to {save_path}...")
            preference_tuples = []

        num_episodes = len(dataset.episodes)
        for n, episode in enumerate(dataset.episodes):
        # for n, episode in enumerate(dataset.episodes[:10]): # For testing
            if n < num_loaded: # Skip already queried episodes
                continue

            print(f"\nEpisode: {n+1}/{num_episodes}")
            episode_preference_tuples = []
            episode_frames = []

            num_obs = len(episode.observations)
            assert num_obs % self.subtrajectory_len == 0, "Subtrajectory length must divide episode length"
            for i in range(0, len(episode.observations), self.subtrajectory_len):
                print(f"  Observation: {i}-{i+self.subtrajectory_len}/{num_obs}")
                sub_obs_video1 = episode.observations[i:i+self.subtrajectory_len]
                
                # randommly sample an episode for comparison
                j = random.randint(0, num_episodes-1)
                # randommly sample a subtrajectory for comparison
                potential_start_indices = list(range(0, len(episode.observations), self.subtrajectory_len))
                potential_start_indices.remove(i)
                k = random.choice(potential_start_indices)

                sub_obs_video2 = dataset.episodes[j].observations[k:k+self.subtrajectory_len]

                # Subsample subtrajectories
                short_sub_obs1 = sub_obs_video1[::self.subsample]
                short_sub_obs2 = sub_obs_video2[::self.subsample]

                # Convert numpy arrays to frames
                render_env = gym.make(self.env_name, render_mode="rgb_array")
                base64_frames_vid1, last_frame = render_trajectory_base64(render_env, short_sub_obs1)
                base64_frames_vid2, last_frame = render_trajectory_base64(render_env, short_sub_obs2)

                # Get VLM preference scores
                pref = self.query_vlm(base64_frames_vid1, base64_frames_vid2)
                # Append preference for i, episode index for comparison, subtrajectory index for comparison               
                episode_preference_tuples.append((pref, j, k)) 

                # Create videos for visualization
                if self.visualize_data:
                    overlay1 = np.zeros_like(last_frame, dtype=np.uint8)
                    overlay2 = np.zeros_like(last_frame, dtype=np.uint8)
                    if pref > 0.5:
                        overlay1[:, :, 1] = 200 # Green tint
                        overlay2[:, :, 0] = 200 # Red tint
                    if pref < 0.5:
                        overlay1[:, :, 0] = 200 # Red tint
                        overlay2[:, :, 1] = 200 # Green tint
                    
                    for obs1, obs2 in zip(sub_obs_video1, sub_obs_video2):
                        # Video 1 frame
                        render_env.reset()
                        render_env.unwrapped.state = np.arctan2(obs1[1], obs1[0]), obs1[2]
                        frame1 = render_env.render()
                        # Video 2 frame
                        render_env.reset()
                        render_env.unwrapped.state = np.arctan2(obs2[1], obs2[0]), obs2[2]
                        frame2 = render_env.render()
                        # Blend frames
                        blended_frame_1 = (0.85 * frame1 + 0.15 * overlay1).astype(np.uint8)
                        blended_frame_2 = (0.85 * frame2 + 0.15 * overlay2).astype(np.uint8)
                        # Stack frames next to each other
                        blended_frame = np.hstack([blended_frame_1, blended_frame_2])
                        episode_frames.append(blended_frame)

            if self.visualize_data:
                # Video directory
                vid_dir = "pref_videos_"+identifier
                # Create directory if it doesn't exist
                os.makedirs(vid_dir, exist_ok=True)
                # Save video
                imageio.mimsave(f"{vid_dir}/sfbc_ep{n+1}.mp4", episode_frames, fps=30)

            preference_tuples.append(episode_preference_tuples)

            # Save confidence scores after each episode in case of interruption
            np.save(save_path, np.array(preference_tuples))

        print(f"Complete VLM preference scores saved at {save_path}")

        return preference_tuples

    def build_with_dataset(self, dataset):
        return # Do later after filtering

    def query_vlm(self, subtrajectory1, subtrajectory2, tries=3):
        """Queries OpenAI VLM for confidence score on a subtrajectory."""
        if tries == 0:
            raise Exception("VLM API Error: Max retries exceeded")
        
        assert SINGLE_SEED, "Only one seed should be used for querying VLM; Multiple is dangerous"
        
        # Construct messages with system prompt + images
        messages = [
            {"role": "system", "content": self.system_prompt},  # Set system prompt
            {"role": "user", "content": [{"type": "text", "text": self.video1_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": self.video2_prompt}]}
        ]

        # Append each base64-encoded image to the user message correctly
        for base64_image in subtrajectory1:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"}  # Correct format
            })
        for base64_image in subtrajectory2:
            messages[2]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"}  # Correct format
            })

        try:
            # OpenAI API call using new client
            response = self.client.chat.completions.create(
                model="gpt-4o", # gpt-4o-mini (does not work well) or gpt-4o
                messages=messages,
                max_tokens=1,  # We only want a 1/2 response
                logprobs=True,  # Ensure we get log probabilities
                temperature=0,  # Make it deterministic
                top_logprobs=5,  # Get logprobs for the top 5 tokens,
            )

        except Exception as e:
            print(f"  VLM API Error: {e}")
            return self.query_vlm(subtrajectory1, subtrajectory2, tries - 1)

        # Extract the most likely token
        choice = response.choices[0]
        predicted_token = choice.message.content.strip().lower()
        print(f"  VLM Response: {predicted_token}")

        # Extract logprobs from the response
        one_prob = 0.0
        two_prob = 0.0

        # Yes/No Variants
        one_variants = {"1", "one",}
        two_variants = {"2", "two",}

        # Extract top log probabilities
        if choice.logprobs:
            for type, logprob_entry in choice.logprobs:
                if type == "content":
                    for token_entry in logprob_entry:
                        token = token_entry.token.strip().lower()
                        prob = np.exp(token_entry.logprob)  # Convert log-prob to prob
                        if token in one_variants:
                            one_prob += prob
                        elif token in two_variants:
                            two_prob += prob

        print(f"  Aggregated 1 Probability: {one_prob}, Aggregated 2 Probability: {two_prob}")

        # Return .5 if probabilities are equal. This also covers the case where both are zero.
        if one_prob == two_prob:
            return 0.5
        
        # Normalize probabilities
        one_prob = one_prob / (one_prob + two_prob)

        assert 0 <= one_prob <= 1, "Invalid probability value"
        return one_prob

class sfbc:
    def __init__(self, learning_rate=1e-3, critic_learning_rate=1e-3, visualize_data=True, 
                 env_name="Pendulum-v1", subsample=20, subtrajectory_len=100,
                 use_vlm_weights=True, strict_filter=True, vlm_confidence_threshold=0.1,
                 td3bc_instead=False, awac_instead=False, sparse_prompt=False): 
        self.subtrajectory_len = subtrajectory_len
        self.vlm_confidence_threshold = vlm_confidence_threshold
        self.use_vlm_weights = use_vlm_weights
        self.bc_agent = None
        self.subsample = subsample
        self.env_name = env_name
        self.awac_instead = awac_instead
        self.td3bc_instead = td3bc_instead
        assert not (self.awac_instead and self.td3bc_instead), "Cannot use both AWAC and TD3+BC"
        self.visualize_data = visualize_data
        self.learning_rate = learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.strict_filter = strict_filter
        self.sparse_prompt = sparse_prompt
        if sparse_prompt:
            self.vlm_prompts = ["You are watching a video of a red stick. If the black dot is at the bottom of the stick, answer 'Y'. Otherwise, answer 'N'.",]
        else:
            self.vlm_prompts = ["You are watching a video of a red stick. If the black dot is at the bottom of the stick, answer 'Y'. Otherwise, answer 'N'.",
                                "You are watching a video of a red stick. If the stick has moved between sides of the screen (left to right or right to left), answer 'Y'. Otherwise, answer 'N'."]
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def fit(self, dataset, **kwargs):
        # Query VLM for confidence scores, if not already done
        confidence_scores = self.get_confidence(dataset)
        # Filter the dataset using VLM scores
        filtered_dataset, unfiltered_dataset = self.filter_dataset(dataset, confidence_scores)
        # Make agent
        if self.awac_instead: 
            # Do AWAC instead
            self.agent = d3rlpy.algos.AWACConfig(actor_learning_rate=self.learning_rate, 
                                                 critic_learning_rate=self.critic_learning_rate).create(device=DEVICE)
        elif self.td3bc_instead:
            # Do TD3+BC instead
            self.agent = d3rlpy.algos.TD3PlusBCConfig(actor_learning_rate=self.learning_rate, 
                                                     critic_learning_rate=self.critic_learning_rate).create(device=DEVICE)
        else:
            # Do BC on filtered data
            if self.use_vlm_weights:
                self.agent = WBCConfig(learning_rate=self.learning_rate).create(device=DEVICE)
            else:
                self.agent = d3rlpy.algos.BCConfig(learning_rate=self.learning_rate).create(device=DEVICE)
        # Build the model with dataset
        self.agent.build_with_dataset(filtered_dataset)
        # Fit the model
        if self.awac_instead:
            return self.agent.fit(unfiltered_dataset, **kwargs)
        elif self.td3bc_instead:
            # Given BC component, you can also filter
            return self.agent.fit(unfiltered_dataset, **kwargs)
        else:
            return self.agent.fit(filtered_dataset, **kwargs)

    def predict(self, observation):
        # Predict the action
        return self.agent.predict(observation)

    def get_confidence(self, dataset):
        """
        Queries VLM for confidence scores on dataset and returns the scores.
        """
        # Compute hash of dataset (using observations & actions)
        dataset_hash = hashlib.md5(np.stack(dataset.episodes[0].observations).tobytes()).hexdigest()
        algo_hash = hashlib.md5(f"{self.subtrajectory_len}_{self.subsample}".encode()).hexdigest()
        combined_hash = hashlib.md5(f"{dataset_hash}_{algo_hash}".encode()).hexdigest()
        identifier = combined_hash
        if self.sparse_prompt:
            identifier = "sparse_" + identifier
        if self.subtrajectory_len == 600:
            identifier = "full_" + identifier
        # Save path for numpty array of confidence scores
        save_path = f"{identifier}_vlm.npy"

        # Check if the confidence scores already exist
        num_loaded = 0
        if os.path.exists(save_path):
            # If the scores are complete, load them instead of re-querying
            loaded_scores = np.load(save_path)
            num_loaded = len(loaded_scores)
            print(f"Found {num_loaded} VLM confidence scores saved at {save_path}")
            if num_loaded == len(dataset.episodes):
                print(f"VLM confidence scores are complete; loading instead.")
                return loaded_scores
            print(f"VLM confidence scores are incomplete; resuming queries...")
            confidence_scores = list(loaded_scores)
        else:
            print(f"Querying VLM for confidence scores and saving to {save_path}...")
            confidence_scores = []

        num_episodes = len(dataset.episodes)
        for n, episode in enumerate(dataset.episodes):
        # for n, episode in enumerate(dataset.episodes[:10]): # For testing
            if n < num_loaded: # Skip already queried episodes
                continue

            print(f"\nEpisode: {n+1}/{num_episodes}")
            episode_confidence_scores = []
            episode_frames = []

            num_obs = len(episode.observations)
            assert num_obs % self.subtrajectory_len == 0, "Subtrajectory length must divide episode length"
            for i in range(0, len(episode.observations), self.subtrajectory_len):
                print(f"  Observation: {i}-{i+self.subtrajectory_len}/{num_obs}")
                sub_obs = episode.observations[i:i+self.subtrajectory_len]

                # Subsample subtrajectory
                short_sub_obs = sub_obs[::self.subsample]

                # Convert numpy arrays to frames
                render_env = gym.make(self.env_name, render_mode="rgb_array")
                base64_frames = []
                for state in short_sub_obs:
                    render_env.reset()
                    render_env.unwrapped.state = np.arctan2(state[1], state[0]), state[2]
                    frame = render_env.render()
                    base64_image = render_frame_to_base64(frame)
                    base64_frames.append(base64_image)

                # Get VLM confidence scores
                vlm_conf = 0.0
                for vlm_prompt in self.vlm_prompts:
                    print(f"  Querying VLM with prompt: {vlm_prompt}")
                    vlm_conf += self.query_vlm(base64_frames, vlm_prompt)
                vlm_conf = min(vlm_conf, 1.0)  # Clip to 1.0
                print(f"  Combined VLM Confidence: {vlm_conf}\n")
                # vlm_conf = random.random()  # For testing
                episode_confidence_scores.append(vlm_conf)                

                # Create videos for visualization
                if self.visualize_data:
                    overlay = np.zeros_like(frame, dtype=np.uint8) # green or red overlay
                    if vlm_conf >= self.vlm_confidence_threshold:
                        overlay[:, :, 1] = 200  # Green tint
                    else:
                        overlay[:, :, 0] = 200  # Red tint
                    for obs in sub_obs:
                        render_env.reset()
                        render_env.unwrapped.state = np.arctan2(obs[1], obs[0]), obs[2]
                        frame = render_env.render()
                        blended_frame = (0.85 * frame + 0.15 * overlay).astype(np.uint8)
                        episode_frames.append(blended_frame)

            if self.visualize_data:
                # Video directory
                vid_dir = "sfbc_videos_"+identifier
                # Create directory if it doesn't exist
                os.makedirs(vid_dir, exist_ok=True)
                # Save video
                imageio.mimsave(f"{vid_dir}/sfbc_ep{n+1}.mp4", episode_frames, fps=30)

            confidence_scores.append(episode_confidence_scores)

            # Save confidence scores after each episode in case of interruption
            np.save(save_path, np.array(confidence_scores))

        print(f"Complete VLM confidence scores saved at {save_path}")

        return confidence_scores

    def filter_dataset(self, dataset, confidence_scores):
        """
        Filters dataset using VLM saved scores and constructs a new MDPDataset.
        Also clamp actions to [-2, 2] for Pendulum environment.

        Returns:
            MDPDataset: Filtered dataset with high-confidence sub-trajectories.
        """
        print(f"Filtering dataset")

        observations, actions, rewards, terminations, truncations = [], [], [], [], []
        unfiltered_rewards, unfiltered_terminations, unfiltered_truncations = [], [], []

        num_episodes = len(dataset.episodes)
        for n, episode in enumerate(dataset.episodes):
        # for n, episode in enumerate(dataset.episodes[:10]): # For testing
            print(f"\nEpisode: {n+1}/{num_episodes}")
            episode_observations, episode_actions, episode_rewards = [], [], []
            episode_terminations, episode_truncations = [], []
            unfiltered_episode_rewards, unfiltered_episode_terminations, unfiltered_episode_truncations = [], [], []

            # Get VLM confidence score
            episode_confidence_scores = confidence_scores[n]

            num_obs = len(episode.observations)
            assert num_obs % self.subtrajectory_len == 0, "Subtrajectory length must divide episode length"
            for i in range(0, len(episode.observations), self.subtrajectory_len):
                print(f"  Observation: {i}-{i+self.subtrajectory_len}/{num_obs}")
                sub_obs = episode.observations[i:i+self.subtrajectory_len]
                sub_act = episode.actions[i:i+self.subtrajectory_len]

                # Get VLM confidence score
                index = i // self.subtrajectory_len
                vlm_conf = episode_confidence_scores[index]

                # Define filtering
                pass_filter = vlm_conf >= self.vlm_confidence_threshold
                if self.strict_filter:
                    # Strict filter: Do not include if next subtrajectory is low confidence
                    if index < len(episode_confidence_scores) - 1:
                        next_vlm_conf = episode_confidence_scores[index + 1]
                        if next_vlm_conf < self.vlm_confidence_threshold:
                            pass_filter = False

                # Filter subtrajectory
                if pass_filter:
                    episode_observations.extend(sub_obs)
                    episode_actions.extend(sub_act)
                    episode_rewards.extend(np.ones(len(sub_obs))*vlm_conf)  # Fill with confidence score
                    episode_terminations.extend([False] * len(sub_obs))
                    episode_truncations.extend([False] * len(sub_obs))
                unfiltered_episode_rewards.extend(np.ones(len(sub_obs))*vlm_conf)
                unfiltered_episode_terminations.extend([False] * len(sub_obs))
                unfiltered_episode_truncations.extend([False] * len(sub_obs))

            if episode_observations:
                episode_truncations[-1] = True  # Mark last step as truncated
                observations.append(episode_observations)
                actions.append(episode_actions)
                rewards.append(episode_rewards)
                terminations.append(episode_terminations)
                truncations.append(episode_truncations)
            unfiltered_episode_truncations[-1] = True  # Mark last step as truncated
            unfiltered_rewards.append(unfiltered_episode_rewards)
            unfiltered_terminations.append(unfiltered_episode_terminations)
            unfiltered_truncations.append(unfiltered_episode_truncations)

        # Convert lists to NumPy arrays
        observations = np.vstack(observations)
        actions = np.vstack(actions)
        rewards = np.hstack(rewards)
        terminations = np.hstack(terminations)
        truncations = np.hstack(truncations)
        unfiltered_rewards = np.hstack(unfiltered_rewards)
        unfiltered_terminations = np.hstack(unfiltered_terminations)
        unfiltered_truncations = np.hstack(unfiltered_truncations)

        # Convert to d3rlpy dataset
        filtered_dataset = MDPDataset(observations, actions, rewards, terminations, truncations)

        # Create an unfiltered version, where rewards are confidence scores
        unfiltered_dataset = MDPDataset(np.vstack([ep.observations for ep in dataset.episodes]), np.vstack([ep.actions for ep in dataset.episodes]), 
                                        unfiltered_rewards, unfiltered_terminations, unfiltered_truncations)

        return filtered_dataset, unfiltered_dataset

    def build_with_dataset(self, dataset):
        return # Do later after filtering

    def query_vlm(self, subtrajectory, vlm_prompt, tries=3):
        """Queries OpenAI VLM for confidence score on a subtrajectory."""
        if tries == 0:
            raise Exception("VLM API Error: Max retries exceeded")
        
        assert SINGLE_SEED, "Only one seed should be used for querying VLM; Multiple is dangerous"
        
        # Construct messages with system prompt + images
        messages = [
            {"role": "system", "content": vlm_prompt},  # Set system prompt
            {"role": "user", "content": []}  # User message starts empty
        ]

        # Append each base64-encoded image to the user message correctly
        for base64_image in subtrajectory:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"}  # Correct format
            })

        try:
            # OpenAI API call using new client
            response = self.client.chat.completions.create(
                model="gpt-4o", # gpt-4o-mini (does not work well) or gpt-4o
                messages=messages,
                max_tokens=1,  # We only want a Yes/No response
                logprobs=True,  # Ensure we get log probabilities
                temperature=0,  # Make it deterministic
                top_logprobs=5,  # Get logprobs for the top 5 tokens,
            )

        except Exception as e:
            print(f"  VLM API Error: {e}")
            return self.query_vlm(subtrajectory, tries - 1)

        # Extract the most likely token
        choice = response.choices[0]
        predicted_token = choice.message.content.strip().lower()
        print(f"  VLM Response: {predicted_token}")

        # Extract logprobs from the response
        yes_prob = 0.0
        no_prob = 0.0

        # Yes/No Variants
        yes_variants = {"y", "yes",}
        no_variants = {"n", "no",}

        # Extract top log probabilities
        if choice.logprobs:
            for type, logprob_entry in choice.logprobs:
                if type == "content":
                    for token_entry in logprob_entry:
                        token = token_entry.token.strip().lower()
                        prob = np.exp(token_entry.logprob)  # Convert log-prob to prob
                        if token in yes_variants:
                            yes_prob += prob
                        elif token in no_variants:
                            no_prob += prob

        print(f"  Aggregated YES Probability: {yes_prob}, Aggregated NO Probability: {no_prob}")

        assert 0 <= no_prob <= 1, "Invalid probability value"
        return 1. - no_prob # No probability is more reliable


def train(seed, lr, critic_lr, data_name="Pendulum_Stitched", algo="awac", 
          vis_data=False, num_steps=500, eval_during_training=True):
    
    assert data_name in ["Pendulum-v1", "Pendulum_Stitched",], "Invalid environment name"
    assert algo in ["awac", "bc", "td3+bc", "sfbc", "dpo"], "Invalid agent name"

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
        # Print information about the dataset
        all_actions = np.vstack([ep.actions for ep in mdp_dataset.episodes])
        print("Min action:", all_actions.min())
        print("Max action:", all_actions.max())
        print("Mean action:", all_actions.mean())
        print("Standard deviation:", all_actions.std())
        # Clip actions to [-2, 2]
        if all_actions.min() < -2 or all_actions.max() > 2:
            print("Clipping actions to [-2, 2]")
            all_actions = np.clip(all_actions, -2, 2)
            # Copy the other data
            all_obs = np.vstack([ep.observations for ep in mdp_dataset.episodes])
            all_rewards = np.hstack([ep.rewards for ep in mdp_dataset.episodes])
            all_terminations = []
            all_truncations = []
            for ep in mdp_dataset.episodes:
                ep_terminations = [False] * len(ep.observations)
                all_terminations.extend(ep_terminations)
                ep_trunc = [False] * len(ep.observations)
                ep_trunc[-1] = True  # Only set the last step as truncated
                all_truncations.extend(ep_trunc)
            all_terminations = np.hstack(all_terminations)
            all_truncations = np.hstack(all_truncations)
            # Convert to 1D array since MDPDataset expects flat truncations
            all_truncations = np.hstack(all_truncations)
            # Create a new MDPDataset
            mdp_dataset = MDPDataset(all_obs, all_actions, all_rewards, all_terminations, all_truncations)
            all_actions = np.vstack([ep.actions for ep in mdp_dataset.episodes])
            assert all_actions.min() >= -2 and all_actions.max() <= 2, "Actions not clamped correctly"

    print("Dataset size:", len(mdp_dataset.episodes))

    # Visualize data
    if vis_data:
        visualize_data(env_name, mdp_dataset, data_name)

    # Initialize model
    if algo == "awac":
        agent = d3rlpy.algos.AWACConfig(actor_learning_rate=lr, 
                                        critic_learning_rate=critic_lr).create(device=DEVICE)
    elif algo == "bc":
        agent = d3rlpy.algos.BCConfig(learning_rate=lr).create(device=DEVICE)
    elif algo == "td3+bc":
        agent = d3rlpy.algos.TD3PlusBCConfig(actor_learning_rate=lr, 
                                             critic_learning_rate=critic_lr).create(device=DEVICE)
    elif algo == "sfbc":
        agent = sfbc(env_name=env_name, learning_rate=lr, critic_learning_rate=critic_lr)
    elif algo == "dpo":
        agent = pref_agent(env_name=env_name, learning_rate=lr)
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
        evaluators={'environment': env_evaluator} if eval_during_training else None, 
        logger_adapter=d3rlpy.logging.NoopAdapterFactory())
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
    if "environment" in history[0][1]:
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

def rollout(seed, agent, env_name, filename, save_rollout_videos, fps=30, max_steps=1000):
    print(f"Generating rollout: {filename}")
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
    done = False
    total_reward = 0.
    step_num = 0
    is_verticals = []
    while not done:
        action = agent.predict(np.array([observation]))[0]  # Get policy action
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        # Determine whether vertical
        cos_theta, sin_theta, _ = observation
        theta = np.arctan2(sin_theta, cos_theta)
        is_verticals.append(1 if abs(theta) <= 0.5 else 0)
        step_num += 1
        if save_rollout_videos:
            frame = env.render()
            if frame is None:
                print("Warning: Environment did not return frames. Skipping video generation.")
                save_rollout_videos = False
            frames.append(frame)
        # print(f"Step: {len(frames)} Reward: {reward} Done: {done}")
        if step_num >= max_steps:
            break

    # Save video
    if save_rollout_videos:
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Saved rollout video at {filename}")

    success = int(sum(is_verticals) >= step_num/2)
    return total_reward, success

def set_seed(seed):
    """Set random seed globally for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gym.utils.seeding.np_random(seed)
    d3rlpy.seed(seed)

def generate_stitched_dataset(seed, dataset_name, n_episodes):
    """Generate dataset where a random half of each episode is expert and the other half is anti-expert."""
    set_seed(seed)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    observations, actions, rewards, terminations, truncations = [], [], [], [], []

    # Note: Starting in a more diverse range of states could help enable more generalization
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

            # Get expert action
            cos_theta, sin_theta, theta_dot = obs
            theta = np.arctan2(sin_theta, cos_theta)
            if theta <= -3 * np.pi / 4 or theta >= 3 * np.pi / 4: # Pendulum is pointed down
                expert_action = np.array([.7]) if theta_dot > 0 else np.array([-.7]) # Pick up momentum
            else:
                expert_action = np.array([-70. * theta - 10. * theta_dot])  # PD control

            # Decide which action to take
            if (expert_first and step < max_steps // 2) or (not expert_first and step >= max_steps // 2):
                # Use Expert for this half
                action = expert_action
            else:
                # Use Anti-Expert
                goal = np.pi if theta > 0 else -np.pi
                if abs(theta - goal) < 0.1:
                    # If at bottom, oppose velocity slightly
                    action = np.array([-5. * theta_dot])  # D control
                elif abs(theta) < 0.15: 
                    # If at top, nudge to the right
                    action = np.array([-0.1])
                elif -7*np.pi/8 < theta < -np.pi/4:
                    # Hard counter clockwise
                    action = np.array([2.0])
                else:
                    # oppose velocity
                    action = np.array([-200. * theta_dot])  # D control
            
            action = np.clip(action, -2, 2)  # Clip to [-2, 2]

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
    visualize_data("Pendulum-v1", mdp_dataset, "Pendulum_Stitched", episodes=[0,1,2,3,4])

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


if __name__ == "__main__":
    seeds = [20, 73, 11, 46, 89, 18, 12, 37, 94, 83, 13, 53, 61, 77, 22,] # 15 seeds # [20] for single seed
    data_name = "Pendulum_Stitched" # "Pendulum-v1", "Pendulum_Stitched"
    algo = "sfbc" # "awac", "bc", "td3+bc", "sfbc", "dpo"
    num_stitched_episodes = 500
    lr = 1e-3 if algo in ["bc", "sfbc", "dpo"] else 3e-4 # defaults from d3rlpy
    critic_learning_rate = lr
    num_step = 8000
    eval_during_training = False
    save_rollout_videos = True

    assert len(sys.argv) == 2, "Usage: python offline.py <openai_api_key or None>"
    OPENAI_API_KEY = sys.argv[1]

    if len(seeds) == 1:
        SINGLE_SEED = True

    # If there is no "Pendulum_Stitched" dataset locally, create it
    if data_name == "Pendulum_Stitched" and not os.path.exists(STITCHED_PATH):
        stitch_seed = 20
        set_seed(stitch_seed)
        print("Generating stitched dataset...")
        generate_stitched_dataset(stitch_seed, STITCHED_PATH, num_stitched_episodes)
        print("Done generating data; please run the script again.")
        exit() # To allow for inspection before calling epensive OpenAI API

    # Train and evaluate
    if len(seeds) > 1:
        # Asynchronous training
        pool = Pool(min(8, len(seeds))) # Too many seeds causes script to hang
        pool_args = [(seed, lr, critic_learning_rate, save_rollout_videos,
                    data_name, algo, num_step, eval_during_training) for seed in seeds]
        returns, successes = zip(*pool.starmap(train_and_eval, pool_args))
        pool.close()
        pool.join()
    else:
        r, succ = train_and_eval(seeds[0], lr, critic_learning_rate, save_rollout_videos,
                                            data_name, algo, num_step, eval_during_training)
        returns = [r]
        successes = [succ]
    print(f"\nDone evaluating {algo}...")
    print("Returns:", returns)
    print("Return Mean:", np.mean(list(returns)))
    print("Return Std:", np.std(list(returns)))
    print("Return Std Err:", np.std(list(returns)) / np.sqrt(len(returns)))
    print("Successes:", successes)
    print("Success Percent:", 100*np.mean(list(successes)))
    print("\n")