import gym
import numpy as np
from gym import spaces
from typing import Callable, Union, Dict, Any, Tuple
from SemiPhysBuildingSim.common.action_transformation import action_index_to_array, array_to_action_index, get_action_mask_fast

# 定义一个类型别名，用于 action mask 生成函数
# 新签名：输入是新的观测值(obs)、上一步的动作(action)和动作空间(action_space)，输出是掩码(mask)
ActionMaskFn = Callable[[Union[np.ndarray, Dict[str, np.ndarray]], int, spaces.Discrete], np.ndarray]


def get_dummy_action_mask(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:
    """
    一个简单的占位符函数，用于生成动作掩码。
    它现在接收新的观测值、上一步的动作和动作空间作为参数。

    :param obs: 环境的观测值 (当前未使用)。
    :param action: 上一步执行的动作 (当前未使用)。在 episode 开始时（即 reset后），该值为 -1。
    :param action_space: 环境的离散动作空间。
    :return: 一个全为 1 的 numpy 数组，形状为 (action_space.n,)，表示所有动作都可用。
    """
    # 打印收到的参数，方便调试 (可注释掉)
    # print(f"Generating mask with last_action: {action}, obs shape: {obs.shape if isinstance(obs, np.ndarray) else 'Dict Obs'}")
    return np.ones(action_space.n, dtype=np.int8)


def create_fixed_action_mask(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:

    # 1. 定义或计算 controllable_rooms
    # 暂时固定：第5个和第7个房间（0-indexed 下的索引4和6）始终不可控。
    # !!! 这里的不可控是指房间所有的action都服从前一个动作
    controllable_rooms = np.array([1, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    # controllable_rooms = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    # TODO: 未来，可以根据 obs 来动态生成 controllable_rooms
    # 例如: controllable_rooms = get_controllable_from_obs(obs)

    old_action_index = last_action

    action_mask = get_action_mask_fast(controllable_rooms, old_action_index, action_space)

    return action_mask


def create_fixed_action_mask_2nd(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:

    # 1. 定义或计算 controllable_rooms
    # 暂时固定：第5个和第7个房间（0-indexed 下的索引4和6）始终不可控。
    # !!! 这里的不可控是指不可控房间的action都是0，即不允许控制
    controllable_rooms = np.array([1, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    # controllable_rooms = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
    # TODO: 未来，可以根据 obs 来动态生成 controllable_rooms
    # 例如: controllable_rooms = get_controllable_from_obs(obs)

    old_action_index = 0

    action_mask = get_action_mask_fast(controllable_rooms, old_action_index, action_space)

    return action_mask


class ActionMasker(gym.Wrapper):
    """
    一个 Gym Wrapper，用于在环境的 `info` 字典中自动添加 'action_mask'。

    它使用一个可配置的函数来根据当前观测值和动作空间生成掩码。
    """

    def __init__(self, env: gym.Env, action_mask_fn: ActionMaskFn = create_fixed_action_mask_2nd):
        """
        初始化 Wrapper。

        :param env: 要包装的 Gym 环境。
        :param action_mask_fn: 一个函数，接收观测值 (obs) 和动作空间 (action_space) 并返回动作掩码 (action_mask)。
        """
        super().__init__(env)

        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError("ActionMasker only supports Discrete action spaces.")

        self.action_mask_fn = action_mask_fn

    def _get_action_mask(self, obs: Union[np.ndarray, Dict[str, np.ndarray]], last_action: int) -> np.ndarray:
        """
        内部辅助函数，用于调用掩码生成函数。
        """
        # 调用时传入 obs, last_action, 和 self.action_space
        return self.action_mask_fn(obs=obs, last_action=last_action, action_space=self.action_space)

    def step(self, action: int) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """
        在环境中执行一步，并在返回的 info 字典中添加下一个状态的 'action_mask'。
        """
        obs, reward, done, info = self.env.step(action)

        action_mask = self._get_action_mask(obs=obs, last_action=action)
        info['action_mask'] = action_mask

        return obs, reward, done, info