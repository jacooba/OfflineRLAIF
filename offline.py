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
import time
import json
import signal
import multiprocessing
matplotlib.use("Agg")  # Use non-interactive backend on Mac

import matplotlib.pyplot as plt

from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import EnvironmentEvaluator
from io import BytesIO
from PIL import Image

# This is not working yet:
# from weighted_bc import RewardWeightedBC

STITCHED_PATH = "Pendulum_Stitched.h5"
EXPERT_PATH = "Pendulum_Expert.d3"
OPENAI_API_KEY = None

def visualize_episode_feedback(args):
    """Visualize an episode with feedback overlay."""
    ep_num, episode, feedback, env_name, vlm_confidence_threshold = args
    render_env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    num_confs = len(feedback)
    len_subtrajectory = len(episode.observations) // num_confs
    for i, vlm_conf in enumerate(feedback):
        sub_trajectory = episode.observations[i:i+len_subtrajectory]
        overlay = np.zeros_like(frame, dtype=np.uint8) # green or red overlay
        if vlm_conf >= vlm_confidence_threshold:
            overlay[:, :, 1] = 200  # Green tint
        else:
            overlay[:, :, 0] = 200  # Red tint
        for obs in sub_trajectory:
            render_env.reset()
            render_env.unwrapped.state = np.arctan2(obs[1], obs[0]), obs[2]
            frame = render_env.render()
            blended_frame = (0.85 * frame + 0.15 * overlay).astype(np.uint8)
            frames.append(blended_frame)
    # Create dir if it doesn't exist
    os.makedirs("sfbc_videos", exist_ok=True)
    imageio.mimsave(f"sfbc_videos/episode_{ep_num}.mp4", frames, fps=30)

def convert_episode_to_frames(args):
    episode, subtrajectory_len, subsample, env_name = args
    num_obs = len(episode.observations)
    assert num_obs % subtrajectory_len == 0, "Subtrajectory length must divide episode length"
    base64_frames = []
    for i in range(0, len(episode.observations), subtrajectory_len):
        print(f"  Observation: {i}-{i+subtrajectory_len}/{num_obs}")
        sub_obs = episode.observations[i:i+subtrajectory_len]

        # Subsample subtrajectory
        short_sub_obs = sub_obs[::subsample]

        # Convert numpy arrays to frames
        render_env = gym.make(env_name, render_mode="rgb_array")
        sub_frames = []
        for state in short_sub_obs:
            render_env.reset()
            render_env.unwrapped.state = np.arctan2(state[1], state[0]), state[2]
            frame = render_env.render()
            base64_image = render_frame_to_base64(frame)
            sub_frames.append(base64_image)
        base64_frames.append(sub_frames)

    return base64_frames

def render_frame_to_base64(frame):
    """Convert a rendered frame (numpy array) to a base64-encoded string."""
    img = Image.fromarray(frame)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class sfbc:
    def __init__(self, subtrajectory_len=100, vlm_confidence_threshold=0.85, visualize_data=True, 
                 use_vlm_weights=True, subsample=25, env_name="Pendulum-v1", awac_instead=False):
        self.subtrajectory_len = subtrajectory_len
        self.vlm_confidence_threshold = vlm_confidence_threshold
        self.use_vlm_weights = use_vlm_weights
        self.bc_agent = None
        self.subsample = subsample
        self.env_name = env_name
        self.awac_instead = awac_instead
        self.visualize_data = visualize_data
        self.vlm_prompt = "The goal is to swing the pendulum back and forth until vertical and then balance the pendulum so it spends as much time vertical as possible. Is the task accomplished well? Answer only 'Y' for yes or 'N' for no, with the single letter and no punctuation."
        # For sparse feedback:
        # self.vlm_prompt = "The goal is to balance the pendulum so it spends as much time vertical as possible. Is the task accomplished well? Answer only 'Y' for yes or 'N' for no, with the single letter and no punctuation."
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def fit(self, dataset, **kwargs):
        # Query VLM for confidence scores, if not already done
        confidence_scores = self.get_confidence(dataset)
        # Batched version, but has to be used with 4o-mini due to API limits:
        # confidence_scores = self.get_confidence_batched(dataset)
        # if self.visualize_data:
        #     self.visualize_data_batched(dataset, confidence_scores)
        # Filter the dataset using VLM scores
        filtered_dataset, unfiltered_dataset = self.filter_dataset(dataset, confidence_scores)
        # Make agent
        if self.awac_instead: 
            # Don't do BC on filtered data; do AWAC instead
            self.agent = d3rlpy.algos.AWACConfig().create(device="mps")
        else:
            # Do BC on filtered data
            if self.use_vlm_weights:
                # Not working yet:
                assert False, "Weighted BC not implemented yet."
                # self.agent = RewardWeightedBC()
            else:
                self.agent = d3rlpy.algos.BCConfig().create(device="mps")
        # Build the model with dataset
        self.agent.build_with_dataset(filtered_dataset)
        # Fit the model
        if self.awac_instead:
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
        # Save path for numpty array of confidence scores
        save_path = f"{combined_hash}_vlm.npy"

        # Check if the confidence scores already exist
        if os.path.exists(save_path):
            print(f"VLM confidence scores found: {save_path}, loading instead of re-querying.")
            return np.load(save_path)
        
        print(f"Querying VLM for confidence scores and saving to {save_path}...")
        confidence_scores = []

        num_episodes = len(dataset.episodes)
        for n, episode in enumerate(dataset.episodes):
        # for n, episode in enumerate(dataset.episodes[:10]): # For testing
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

                # Get VLM confidence score
                vlm_conf = self.query_vlm(base64_frames)
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
                # Create directory if it doesn't exist
                os.makedirs("sfbc_videos", exist_ok=True)
                # Save video
                imageio.mimsave(f"sfbc_videos/sfbc_ep{n+1}.mp4", episode_frames, fps=30)

            confidence_scores.append(episode_confidence_scores)

        # Convert lists to NumPy arrays
        confidence_scores = np.array(confidence_scores)

        # Save confidence scores
        np.save(save_path, confidence_scores)
        print(f"VLM confidence scores saved at {save_path}")

        return confidence_scores
    
    def visualize_data_batched(self, dataset, confidence_scores):
        print(f"Visualizing data with VLM feedback...")
        pool = multiprocessing.Pool()
        args = [(n, ep, confidence_scores[n], self.env_name, self.vlm_confidence_threshold) 
                for n, ep in enumerate(dataset.episodes)]
        pool.map(visualize_episode_feedback, args)

    def get_confidence_batched(self, dataset):
        """
        Queries VLM for confidence scores on dataset and returns the scores.
        """
        # Compute hash of dataset (using observations & actions)
        dataset_hash = hashlib.md5(np.stack(dataset.episodes[0].observations).tobytes()).hexdigest()
        algo_hash = hashlib.md5(f"{self.subtrajectory_len}_{self.subsample}".encode()).hexdigest()
        combined_hash = hashlib.md5(f"{dataset_hash}_{algo_hash}".encode()).hexdigest()
        # Save path for numpty array of confidence scores
        save_path = f"{combined_hash}_vlm.npy"

        # Check if the confidence scores already exist
        if os.path.exists(save_path):
            print(f"VLM confidence scores found: {save_path}, loading instead of re-querying.")
            return np.load(save_path)
        
        if self.visualize_data:
            print(f"Warning: Batched confidenc query will not visualize data.")

        # Convert dataset to subtrajectories of frames
        print(f"Converting episodes to frames")
        num_episodes = len(dataset.episodes)

        # Convert in parallel
        pool = multiprocessing.Pool()
        args = [(ep, self.subtrajectory_len, self.subsample, self.env_name) for ep in dataset.episodes]
        sub_trajectories_by_episode = pool.map(convert_episode_to_frames, args)
        pool.close()

        # Flatten list of lists
        sub_trajectories_by_frames = [frames for episode in sub_trajectories_by_episode for frames in episode]

        # Query VLM for confidence scores
        print(f"Querying VLM for confidence scores...")
        # sub_trajectories_by_frames = sub_trajectories_by_frames[0:1] # For testing
        assert len(sub_trajectories_by_frames) == 1, "Only one episode for testing"
        confidence_scores = self.query_vlm_batch(sub_trajectories_by_frames)

        # Reshape confidence scores to match dataset structure
        confidence_scores = np.array(confidence_scores).reshape(num_episodes, -1)

        # Convert lists to NumPy arrays
        confidence_scores = np.array(confidence_scores)

        # Save confidence scores
        np.save(save_path, confidence_scores)
        print(f"VLM confidence scores saved at {save_path}")

        return confidence_scores

    def filter_dataset(self, dataset, confidence_scores):
        """
        Filters dataset using VLM saved scores and constructs a new MDPDataset.

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

                # Filter subtrajectory based on VLM confidence
                if vlm_conf >= self.vlm_confidence_threshold:
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

    def query_vlm(self, subtrajectory, tries=3):
        """Queries OpenAI VLM for confidence score on a subtrajectory."""
        if tries == 0:
            return 1.0  # Fallback confidence if VLM fails
        
        # Construct messages with system prompt + images
        messages = [
            {"role": "system", "content": self.vlm_prompt},  # Set system prompt
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
            print(f"VLM API Error: {e}")
            return self.query_vlm(subtrajectory, tries - 1)

        # Extract the most likely token
        choice = response.choices[0]
        predicted_token = choice.message.content.strip().lower()
        print(f"VLM Response: {predicted_token}")

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

        print(f"VLM Response: {predicted_token}")
        print(f"Aggregated YES Probability: {yes_prob}, Aggregated NO Probability: {no_prob}")

        if yes_prob == 0.0 and no_prob == 0.0:  # If VLM didn't return useful information
            print("VLM did not return a yes or no probability. Retrying...")
            return self.query_vlm(subtrajectory, tries - 1)
        
        # Rewweight yes and no to sum to 1
        yes_prob /= (yes_prob + no_prob)

        print(f"Final YES Probability: {yes_prob}")

        assert 0 <= yes_prob <= 1, "Invalid probability value"
        return yes_prob
    
    def query_vlm_batch(self, batched_subtrajectories):
        """
        Queries OpenAI VLM using batch API for multiple sub-trajectories at once.
        1. Creates a JSONL batch file.
        2. Submits the batch request.
        3. Waits for completion.
        4. Fetches results and extracts confidence scores.

        Args:
            batched_subtrajectories (list of list of str): 
                Each sub-list is a subtrajectory, containing base64-encoded images.

        Returns:
            List of confidence scores (float) for each subtrajectory.
        """

        # **Step 1: Create the JSONL Batch File**
        print(f"Creating batch file for {len(batched_subtrajectories)} sub-trajectories...")

        batch_file_path = "vlm_batch_requests.jsonl"
        with open(batch_file_path, "w") as f:
            for i, subtrajectory in enumerate(batched_subtrajectories):
                messages = [
                    {"role": "system", "content": self.vlm_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": "low"}} for img in subtrajectory
                    ]}
                ]
                request = {
                    "custom_id": f"request-{i+1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini", # 4o batch limits per day are too low
                        "messages": messages,
                        "max_tokens": 1,
                        "logprobs": True,
                        "temperature": 0,
                        "top_logprobs": 5
                    }
                }
                f.write(json.dumps(request) + "\n")

        print(f"Batch file {batch_file_path} created successfully!")

        # **Step 2: Upload the Batch File to OpenAI**
        print("Uploading batch file to OpenAI...")
        batch_input_file = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )
        batch_file_id = batch_input_file.id

        # **Step 3: Submit the Batch Request**
        print(f"Submitting batch job with file ID: {batch_file_id}...")
        batch_response = self.client.batches.create(
            input_file_id=batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Batch VLM evaluation"}
        )

        batch_id = batch_response.id
        print(f"Batch submitted successfully! Batch ID: {batch_id}")

        # **Define the signal handler inside this function**
        def cancel_batch_and_exit(signum, frame):
            print(f"\nReceived signal {signum}. Canceling batch {batch_id} before exiting...")
            try:
                response = self.client.batches.cancel(batch_id)
                if response.status == "cancelling":
                    print(f"Batch {batch_id} is being cancelled.")
                else:
                    print(f"Failed to cancel batch {batch_id}. Current status: {response.status}")
            except Exception as e:
                print(f"Error while cancelling batch: {e}")
            
            sys.exit(1)  # Exit with error status
        # Register signal handlers for program termination
        signal.signal(signal.SIGINT, cancel_batch_and_exit)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, cancel_batch_and_exit) # Handle termination signal

        # **Step 4: Wait for the Batch Job to Complete**
        print(f"Waiting for batch {batch_id} to complete...")
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            if batch_status.status == "completed":
                print(f"Batch {batch_id} completed successfully!")
                output_file_id = batch_status.output_file_id
                break
            elif batch_status.status == "failed":
                raise RuntimeError(f"Batch {batch_id} failed!")
            elif batch_status.status in {"in_progress", "pending"}:
                print("Batch still processing... waiting 1 second.")
                time.sleep(1)

        # **Step 5: Fetch Results**
        print(f"Fetching batch results from file ID: {output_file_id}...")
        file_response = self.client.files.content(output_file_id)
        results = [json.loads(line) for line in file_response.text.split("\n") if line.strip()]

        confidences = []
        for result in results:
            choice = result['response']['body']['choices'][0]
            predicted_token = choice["message"]["content"].strip().lower()
            yes_prob = 0.0
            no_prob = 0.0

            yes_variants = {"y", "yes"}
            no_variants = {"n", "no"}

            for token_entry in choice["logprobs"]["content"]:
                token = token_entry["token"].strip().lower()
                prob = np.exp(token_entry["logprob"])  # Convert log-prob to prob
                if token in yes_variants:
                    yes_prob += prob
                elif token in no_variants:
                    no_prob += prob

            if yes_prob == 0.0 and no_prob == 0.0:
                yes_prob = 0.0  # Default to assuming negative if uncertain
            else:
                yes_prob /= (yes_prob + no_prob)

            confidences.append(yes_prob)

            print(f"VLM Response: {predicted_token}")
            print(f"Aggregated YES Probability: {yes_prob}, Aggregated NO Probability: {no_prob}")

        print(f"Processed {len(confidences)} sub-trajectories!")
        return confidences
            

def train(seed, data_name="Pendulum_Stitched", algo="awac", vis_data=False, num_steps=500):
    assert data_name in ["Pendulum-v1", "Pendulum_Stitched", "antmaze-medium-play-v0"], "Invalid environment name"
    assert algo in ["awac", "bc", "td3+bc", "sfbc"], "Invalid agent name"

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
    elif algo == "sfbc":
        agent = sfbc(env_name=env_name)
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

def generate_stitched_dataset(seed, dataset_name, n_episodes):
    """Generate dataset where a random half of each episode is expert and the other half is anti-expert."""
    set_seed(seed)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")

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
                # Use Anti-Expert (inverted actions)
                action = -10 * expert_action

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
    visualize_data("Pendulum-v1", mdp_dataset, "Pendulum_Stitched", episodes=[0,1,2,3,4]) # episodes=[0, 1, 2, 3, 250, 499])

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
    algo="sfbc"
    num_stitched_episodes = 500

    assert len(sys.argv) == 2, "Usage: python offline.py <openai_api_key or None>"
    OPENAI_API_KEY = sys.argv[1]

    # If there is no "Pendulum_Stitched" dataset locally, create it
    if data_name == "Pendulum_Stitched" and not os.path.exists(STITCHED_PATH):
        stitch_seed = 20
        set_seed(stitch_seed)
        print("Generating stitched dataset...")
        generate_stitched_dataset(stitch_seed, STITCHED_PATH, num_stitched_episodes)
        print("Done generating data; please run the script again.")
        exit()

    set_seed(seed)

    # Train
    history, agent, env, env_name = train(seed, data_name=data_name, algo=algo, vis_data=False)
    # Plot
    plot(history, f"{algo}_plot_{data_name}_s{seed}.png")
    # Rollout
    rollout(seed, agent, env_name, f"{algo}_trained_{data_name}_s{seed}.mp4")