import dataclasses
import torch

from torch.optim import Optimizer
from typing import Union, Callable
from abc import ABCMeta, abstractmethod

from d3rlpy.models.torch.policies import (
    DeterministicPolicy,
    NormalPolicy,
    Policy,
)
from d3rlpy.types import Shape, TorchObservation
from d3rlpy.torch_utility import Modules
from d3rlpy.optimizers import OptimizerWrapper
from d3rlpy.algos.qlearning.torch.bc_impl import BCBaseImpl
from d3rlpy.algos.qlearning.base import QLearningAlgoBase, QLearningAlgoImplBase
from d3rlpy.algos.qlearning.bc import BCConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.models.builders import create_deterministic_policy
from d3rlpy.optimizers.optimizers import make_optimizer_field, OptimizerFactory
from d3rlpy.models.encoders import EncoderFactory, make_encoder_field
from d3rlpy.base import DeviceArg, LearnableConfig
from d3rlpy.torch_utility import CudaGraphWrapper, Modules, TorchMiniBatch
from d3rlpy.dataclass_utils import asdict_as_float


@dataclasses.dataclass(frozen=True)
class ImitationLoss:
    loss: torch.Tensor

@dataclasses.dataclass(frozen=True)
class BCBaseModules(Modules):
    optim: OptimizerWrapper

@dataclasses.dataclass(frozen=True)
class BCModules(BCBaseModules):
    imitator: Union[DeterministicPolicy, NormalPolicy]

def compute_deterministic_weighted_imitation_loss(
    policy: DeterministicPolicy, x: TorchObservation, action: torch.Tensor, weights: torch.Tensor,
) -> ImitationLoss:
    weighted_loss = (policy(x).squashed_mu - action)**2 * weights
    return ImitationLoss(loss=weighted_loss.mean())

@dataclasses.dataclass()
class WBCConfig(LearnableConfig):
    r"""Config of Behavior Cloning algorithm.

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [(a_t - \pi_\theta(s_t))^2]

    Args:
        learning_rate (float): Learing rate.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        batch_size (int): Mini-batch size.
        policy_type (str): the policy type. Available options are
            ``['deterministic', 'stochastic']``.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 100
    learning_rate: float = 1e-3
    policy_type: str = "deterministic"
    optim_factory: OptimizerFactory = make_optimizer_field()
    encoder_factory: EncoderFactory = make_encoder_field()

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "WBC":
        return WBC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "wbc"

class WBC(QLearningAlgoBase[BCBaseImpl, BCConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        assert self._config.policy_type == "deterministic"
        imitator = create_deterministic_policy(
            observation_shape,
            action_size,
            self._config.encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        optim = self._config.optim_factory.create(
            imitator.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = BCModules(optim=optim, imitator=imitator)

        self._impl = WBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            policy_type=self._config.policy_type,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
    
class BCBaseImpl(QLearningAlgoImplBase, metaclass=ABCMeta):
    _modules: BCBaseModules
    _compute_imitator_grad: Callable[[TorchMiniBatch], ImitationLoss]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCBaseModules,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._compute_imitator_grad = (
            CudaGraphWrapper(self.compute_imitator_grad)
            if compiled
            else self.compute_imitator_grad
        )

    def compute_imitator_grad(self, batch: TorchMiniBatch) -> ImitationLoss:
        self._modules.optim.zero_grad()
        loss = self.compute_loss(batch.observations, batch.actions, batch.rewards)
        loss.loss.backward()
        return loss

    def update_imitator(self, batch: TorchMiniBatch) -> dict[str, float]:
        loss = self._compute_imitator_grad(batch)
        self._modules.optim.step()
        return asdict_as_float(loss)
    
    @abstractmethod
    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor, weight_t: torch.Tensor
    ) -> ImitationLoss:
        pass

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        return self.update_imitator(batch)

class WBCImpl(BCBaseImpl):
    _modules: BCModules
    _policy_type: str

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCModules,
        policy_type: str,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            compiled=compiled,
            device=device,
        )
        self._policy_type = policy_type

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._modules.imitator(x).squashed_mu

    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor, weight_t: torch.Tensor
    ) -> ImitationLoss:
        assert self._policy_type == "deterministic"
        assert isinstance(self._modules.imitator, DeterministicPolicy)
        return compute_deterministic_weighted_imitation_loss(
            self._modules.imitator, obs_t, act_t, weight_t,
        )

    @property
    def policy(self) -> Policy:
        return self._modules.imitator

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.optim.optim