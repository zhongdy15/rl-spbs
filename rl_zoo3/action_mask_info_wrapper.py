import gym
import numpy as np
from gym import spaces
from typing import Callable, Union, Dict, Any, Tuple
from SemiPhysBuildingSim.common.action_transformation import action_index_to_array, array_to_action_index, get_action_mask_fast
from llm_baseline_prompt.llm_chat import llm_chat
import json_repair
import configparser
from interpret_obs import interpret_obs, get_current_fan_speed_from_obs

# 定义一个类型别名，用于 action mask 生成函数
# 新签名：输入是新的观测值(obs)、上一步的动作(action)和动作空间(action_space)，输出是掩码(mask)
ActionMaskFn = Callable[[Union[np.ndarray, Dict[str, np.ndarray]], int, spaces.Discrete], np.ndarray]

print("===================== llm config loading =====================")
config = configparser.ConfigParser()
config.read('llm_baseline_prompt/config/config.ini', encoding='utf-8')
test_model_key = config['llm_api']['model'].split('/')[-1]

with open('llm_baseline_prompt/zero_shot_prompt_action_reduce.txt', 'r', encoding='utf-8') as f:
    zero_shot_prompt_template = f.read()

print(f"===================== llm config loading: {test_model_key} =====================")

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
    # !!! 这里的不可控是指不可控房间的action都是0，即不允许控制
    controllable_rooms = np.array([1, 1, 1, 1, 0, 1, 0], dtype=np.int8)

    old_action_index = 0

    action_mask = get_action_mask_fast(controllable_rooms, old_action_index, action_space)

    # action_mask = np.ones(action_space.n, dtype=np.int8)

    return action_mask


def get_confidence_scores_from_llm(obs: np.ndarray, prompt_template: str, config: configparser.ConfigParser) -> np.ndarray:
    """
    集成所有步骤：解释 obs -> 构造 prompt -> 调用 LLM -> 解析 confidence_scores。
    """
    # 1. 解释 obs
    status_text = interpret_obs(obs)

    # 2. 构造完整 Prompt
    full_prompt = prompt_template.replace("[Status Need Replacement]", status_text)

    # 3. 准备调用 LLM
    llm_api_key = config['llm_api']['api_key']
    llm_base_url = config['llm_api']['url']
    llm_model = config['llm_api']['model']

    messages = [
        # prompt.txt 的内容更适合作为 user prompt，因为它直接给出了任务指令
        {"role": "user", "content": full_prompt}
    ]

    # 4. 调用 LLM 并获取响应
    print("===================== Sending Prompt to LLM =====================")
    # print(full_prompt)
    print("=================================================================")

    response_str = llm_chat(llm_api_key, llm_model, llm_base_url, messages)

    # print(f"LLM Response (raw): {response_str}")

    # 5. 解析 LLM 响应
    try:
        response_dict = json_repair.loads(response_str)
        confidence_scores = response_dict["confidence_scores"]
        # 按照 room_1 到 room_7 的顺序提取confidence_scores
        confidence_list = [confidence_scores[i] for i in range(7)]
        confidence_scores = np.array(confidence_list)
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Using default confidence_scores [0,0,0,0,0,0,0].")
        confidence_scores = np.zeros(7)

    try:
        action_dict = json_repair.loads(response_str)
        reason = action_dict["explanation_and_thought_process"]
    except Exception as e:
        print(f"Error parsing LLM response: {e}. Using default reason 'No reason provided'.")
        reason = "No reason provided"

    return confidence_scores, reason


def get_action_mask_from_llm(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    last_action: int,
    action_space: spaces.Discrete
) -> np.ndarray:
    """
    从 LLM 中获取动作掩码。

    :param obs: 环境的观测值。
    :param last_action: 上一步执行的动作。
    :param action_space: 环境的离散动作空间。
    :return: 从 LLM 中获取的动作掩码。
    """

    old_action = get_current_fan_speed_from_obs(obs)
    old_action_index = array_to_action_index(old_action, [4, 4, 4, 4, 4, 4, 4])

    # 从 LLM 获取confidence_scores
    confidence_scores, reason = get_confidence_scores_from_llm(obs, zero_shot_prompt_template, config)
    # 从confidence_scores挑出最大的3个confidence score，该处是controllable，其他位置不controllable
    max_indices = np.argsort(confidence_scores)[-6:]

    # controllable_rooms = np.array([1, 0, 1, 0, 1, 0, 0])
    controllable_rooms = np.zeros_like(old_action, dtype=int)
    controllable_rooms[max_indices] = 1
    print("controllable_rooms:", controllable_rooms)

    # 得到 action mask
    action_mask = get_action_mask_fast(controllable_rooms, old_action_index, action_space)

    return action_mask




class ActionMasker(gym.Wrapper):
    """
    一个 Gym Wrapper，用于在环境的 `info` 字典中自动添加 'action_mask'。

    它使用一个可配置的函数来根据当前观测值和动作空间生成掩码。
    """

    def __init__(self, env: gym.Env, action_mask_fn: ActionMaskFn = get_action_mask_from_llm):
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