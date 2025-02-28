# SFO: Piloting VLM Feedback for Offline RL

This repository contains the implementation of **Sub-Trajectory Filtered Behavior Cloning (SFBC)**, a method that leverages **vision-language model (VLM) feedback** to improve offline reinforcement learning (RL). SFBC filters and weights sub-trajectories based on VLM-derived success probabilities, enabling effective policy learning in the **absence of explicit rewards**.

## Installation
To set up the environment, use Conda:
```bash
conda env create -f environment.yml
conda activate d3rl
```

## Running the Code
To train and evaluate SFBC, simply run:
```bash
python offline.py <openai_api_key>
```

All key arguments and hyperparameters can be modified in `Main.py`. Currently, they are defined as **constants at the top of `Main.py`**, so you should edit them directly before running the script.

## Citing this Work
If you use this code in your research, please cite our paper:

```bibtex
@article{beck2025sfo,
  author    = {Jacob Beck},
  title     = {SFO: Piloting VLM Feedback for Offline RL},
  journal   = {arXiv},
  year      = {2025}
}
```

## Acknowledgments
This implementation is based on **D3RLpy** for offline RL baselines and uses **GPT-4o** as the vision-language model (VLM) for sub-trajectory evaluation.
