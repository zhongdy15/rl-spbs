import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import ReplayBuffer
from mask_DQN.replay_buffer.MaskableReplayBufferSamples import MaskableReplayBufferSamples


class MaskableReplayBuffer(ReplayBuffer):
    """
    一个严格遵循 Stable-Baselines3 v1.7.0 ReplayBuffer 逻辑的 Buffer，增加了存储和采样 action_mask 的功能。

    此版本的 'add' 方法显式接收 action_mask 和 next_action_mask 作为参数。
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

        assert isinstance(action_space, spaces.Discrete), "Action Masking is only supported for Discrete action spaces."

        action_dim = action_space.n

        self.action_masks = np.zeros((self.buffer_size, self.n_envs, action_dim), dtype=np.float32)

        if self.optimize_memory_usage:
            self.next_action_masks = None
        else:
            self.next_action_masks = np.zeros((self.buffer_size, self.n_envs, action_dim), dtype=np.float32)

        # 内存占用警告
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            # 加上我们新增的 buffer 的内存占用
            total_memory_usage += self.action_masks.nbytes
            if self.next_action_masks is not None:
                total_memory_usage += self.next_action_masks.nbytes

            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            action_mask: np.ndarray,
            next_action_mask: np.ndarray,
    ) -> None:
        """
        向 Buffer 中添加新的经验。

        :param action_mask: 当前状态的动作掩码，形状 (n_envs, action_dim)
        :param next_action_mask: 下一个状态的动作掩码，形状 (n_envs, action_dim)
        """
        # Reshape for discrete observations (from original ReplayBuffer)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Reshape for actions (from original ReplayBuffer)
        action = action.reshape((self.n_envs, self.action_dim))

        # --- 存储掩码 ---
        # 直接使用传入的参数，无需从 infos 解析
        self.action_masks[self.pos] = np.array(action_mask).copy()

        if self.optimize_memory_usage:
            self.action_masks[(self.pos + 1) % self.buffer_size] = np.array(next_action_mask).copy()
        else:
            self.next_action_masks[self.pos] = np.array(next_action_mask).copy()

        # --- 存储标准数据（来自原版 ReplayBuffer 的逻辑）---
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        # 更新指针
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> MaskableReplayBufferSamples:
        # 这个方法的逻辑与上一版完全相同，因为数据存储结构没有变
        # ... (此处代码与上一版完全相同，为简洁省略，可以直接复制粘贴) ...
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            next_action_masks = self.action_masks[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            next_action_masks = self.next_action_masks[batch_inds, env_indices, :]

        action_masks = self.action_masks[batch_inds, env_indices, :]
        observations = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        actions = self.actions[batch_inds, env_indices, :]
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)
        rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)

        observations, actions, next_obs, dones, rewards, action_masks, next_action_masks = tuple(map(
            self.to_torch,
            (observations, actions, next_obs, dones, rewards, action_masks, next_action_masks)
        ))

        return MaskableReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=rewards,
            action_masks=action_masks,
            next_action_masks=next_action_masks,
        )

