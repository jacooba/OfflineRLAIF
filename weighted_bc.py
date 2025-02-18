import dataclasses
import torch
from d3rlpy.base import LearnableBase, register_learnable
from d3rlpy.base import LearnableConfig
from d3rlpy.optimizers import AdamFactory
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.models.torch.bc_impl import BCImpl


class RewardWeightedBCImpl(BCImpl):
    """Custom BC implementation that weights the loss using rewards."""

    def _compute_actor_loss(self, batch: TransitionMiniBatch) -> torch.Tensor:
        """Compute loss for Reward-Weighted BC by weighting the loss with rewards."""
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards  # Use rewards as weights

        # Predict actions using the policy
        pred_actions = self.policy(observations)

        # Compute weighted loss (higher reward = higher weight)
        mse_loss = ((pred_actions - actions) ** 2).mean(dim=1)  # Per-sample loss
        weighted_loss = (mse_loss * rewards).mean()  # Apply reward weighting

        return weighted_loss


@dataclasses.dataclass
class RewardWeightedBCConfig(LearnableConfig):
    """Configuration for Reward-Weighted BC."""
    learning_rate: float = 3e-4

    def create(self, device="auto") -> "RewardWeightedBC":
        return RewardWeightedBC(self, device=device)


@register_learnable
class RewardWeightedBC(LearnableBase):
    def __init__(self, config: RewardWeightedBCConfig, device="auto"):
        super().__init__(config, device)
        self.config = config

    def create_impl(self, observation_shape, action_size):
        """Initialize the BC implementation with reward weighting."""
        self.impl = RewardWeightedBCImpl(
            observation_shape,
            action_size,
            AdamFactory(lr=self.config.learning_rate),
            self.device
        )
        self.impl.build()