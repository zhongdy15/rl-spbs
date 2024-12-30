from typing import Any, Dict, List, Optional, Type

import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from hgqn.hypergraph import get_hypergraph_nvec, get_argmax_from_q_values, revert_action


class HGQNNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        hypergraph: List = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        if hypergraph is None:
            raise ValueError("Hypergraph must be provided for HGQNNetwork")

        print("Creating HGQNNetwork")
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.hypergraph = hypergraph

        action_nvec = self.action_space.nvec
        # Dueling Q-Network
        # reference: https://github.com/MoMe36/BranchingDQN/blob/master/model.py

        # Learning To Represent Action Values As a Hypergraph on the Action Vertices
        # reference: https://arxiv.org/abs/2010.14680

        hypergraph_nvec = get_hypergraph_nvec(action_nvec, hypergraph)
        print("Hypergraph nvec:", hypergraph_nvec)

        shared_reprentation = create_mlp(self.features_dim, -1, self.net_arch, self.activation_fn)
        self.shared_reprentation = nn.Sequential(*shared_reprentation)

        self.value_head = nn.Linear(self.net_arch[-1], 1)
        self.adv_heads = nn.ModuleList([nn.Linear(self.net_arch[-1], ac_dim) for ac_dim in hypergraph_nvec])
        # self.adv_heads = nn.ModuleList([nn.Linear(self.net_arch[-1], ac_dim) for ac_dim in action_nvec])

        # action_dim = self.action_space.n  # number of actions
        # q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        # self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        # return self.q_net(self.extract_features(obs, self.features_extractor))
        out = self.shared_reprentation(self.extract_features(obs, self.features_extractor))
        value = self.value_head(out)
        # advs = th.stack([l(out) for l in self.adv_heads], dim=1)

        q_val_list = []
        for l in self.adv_heads:
            advs = l(out)
            q_val = value + advs - advs.mean(-1, keepdim=True)
            q_val_list.append(q_val)

        return q_val_list


    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # q_values = self(observation)
        # # Greedy action
        # action = q_values.argmax(dim=1).reshape(-1)

        q_val_list = self(observation)
        argmax = get_argmax_from_q_values(q_val_list)
        action = revert_action(argmax, self.hypergraph, self.action_space.nvec)

        return action

    def _predict_with_disabled_action(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # q_values = self(observation)
        # # Greedy action
        # action = q_values.argmax(dim=1).reshape(-1)

        q_val_list = self(observation)
        q_val_list[3][0, [2, 3, 6, 7]] = float('-inf')
        argmax = get_argmax_from_q_values(q_val_list)
        action = revert_action(argmax, self.hypergraph, self.action_space.nvec)

        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                hypergraph=self.hypergraph,
            )
        )
        return data


class HGQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        hypergraph: List = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        print("Creating HGQNPolicy")
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        if hypergraph is None:
            raise ValueError("Hypergraph must be provided for HGQNPolicy")

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.hypergraph = hypergraph

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "hypergraph": self.hypergraph,
        }

        self.q_net, self.q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_q_net(self) -> HGQNNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return HGQNNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                hypergraph=self.net_args["hypergraph"],
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode


# MlpPolicy = DQNPolicy
#
#
# class CnnPolicy(DQNPolicy):
#     """
#     Policy class for DQN when using images as input.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param features_extractor_class: Features extractor to use.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[List[int]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )
#
#
# class MultiInputPolicy(DQNPolicy):
#     """
#     Policy class for DQN when using dict observations as input.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param features_extractor_class: Features extractor to use.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[List[int]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )