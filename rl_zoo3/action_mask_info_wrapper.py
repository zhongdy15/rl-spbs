import gym
import numpy as np
from gym import spaces
from typing import Callable, Union, Dict, Any, Tuple

# 定义一个类型别名，用于 action mask 生成函数
# 新签名：输入是观测值(obs)和动作空间(action_space)，输出是掩码(mask)
ActionMaskFn = Callable[[Union[np.ndarray, Dict[str, np.ndarray]], spaces.Discrete], np.ndarray]


def get_dummy_action_mask(obs: Union[np.ndarray, Dict[str, np.ndarray]], action_space: spaces.Discrete) -> np.ndarray:
    """
    一个简单的占位符函数，用于生成动作掩码。
    它现在接收 action_space 参数，并用它来确定掩码的维度。

    :param obs: 环境的观测值 (当前未使用)。
    :param action_space: 环境的离散动作空间。
    :return: 一个全为 1 的 numpy 数组，形状为 (action_space.n,)。
    """
    return np.ones(action_space.n, dtype=np.int8)


class ActionMasker(gym.Wrapper):
    """
    一个 Gym Wrapper，用于在环境的 `info` 字典中自动添加 'action_mask'。

    它使用一个可配置的函数来根据当前观测值和动作空间生成掩码。
    """

    def __init__(self, env: gym.Env, action_mask_fn: ActionMaskFn = get_dummy_action_mask):
        """
        初始化 Wrapper。

        :param env: 要包装的 Gym 环境。
        :param action_mask_fn: 一个函数，接收观测值 (obs) 和动作空间 (action_space) 并返回动作掩码 (action_mask)。
        """
        super().__init__(env)

        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("ActionMasker only supports Discrete action spaces.")

        self.action_mask_fn = action_mask_fn

    def _get_action_mask(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        内部辅助函数，用于调用掩码生成函数。
        """
        # 调用时传入 self.action_space
        return self.action_mask_fn(obs, self.action_space)

    # def reset(self, **kwargs) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Dict[str, Any]]:
    #     """
    #     重置环境，并在返回的 info 字典中添加 'action_mask'。
    #     """
    #     try:
    #         obs, info = self.env.reset(**kwargs)
    #     except (TypeError, ValueError):  # ValueError for some gym versions
    #         obs = self.env.reset(**kwargs)
    #         info = {}
    #
    #     action_mask = self._get_action_mask(obs)
    #     info['action_mask'] = action_mask
    #
    #     return obs, info

    def step(self, action: int) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """
        在环境中执行一步，并在返回的 info 字典中添加下一个状态的 'action_mask'。
        """
        obs, reward, done, info = self.env.step(action)

        action_mask = self._get_action_mask(obs)
        info['action_mask'] = action_mask

        return obs, reward, done, info