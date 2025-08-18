import SemiPhysBuildingSim
import gym
import numpy as np
import matplotlib.pyplot as plt
from rl_zoo3.wrappers import FrameSkip, DisabledWrapper
import os
import re
from openai import OpenAI

# ==============================================================================
# OpenAI Client Initialization
# ==============================================================================
try:
    client = OpenAI(
        api_key="not-needed-for-local",
        base_url="http://localhost:8000/v1"
    )
    print("Successfully initialized OpenAI client for local server.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# ==============================================================================
# Data Keys for Prompt Generation (as per your specification)
# ==============================================================================
# Define the specific keys to be included in the input and output
room_input_keys = [
    "room_temp",  # Room temperature
    "FCU_fan_feedback",  # FCU fan speed feedback
    "supply_temp",  # FCU supply water temperature
    "return_temp",  # FCU return water temperature
    "occupant_num",  # Number of occupants
]
outdoor_input_key = "outdoor_temp"  # Only outdoor temperature
room_output_key = "FCU_fan_setpoint"  # Only FCU fan setpoint

# ==============================================================================
# LLM Interaction Functions
# ==============================================================================

def state2language(env: gym.Env, current_step: int) -> str:
    """
    Generates a detailed prompt by reading the latest data directly from the
    environment's data_recorder.

    Args:
        env: The gym environment instance, used to access env.data_recorder.
        current_step: The current timestep of the simulation loop (starts at 1).
    """
    recorder = env.data_recorder
    t = current_step

    try:
        # Build the input string using the logic from your reference function
        input_str = f"Timestamp index: {t} (August 9th, {9 + t // 60}:{(t % 60):02d}am/pm).\n"
        input_str += "Current building conditions:\n"

        # Outdoor temperature
        input_str += "--- Outdoor Environment Status ---\n"
        temp = recorder["sensor_outdoor"].get(outdoor_input_key, [None])[t]
        input_str += f"- Outdoor Temperature: {temp:.2f}°C\n\n" if temp is not None else "- Outdoor Temperature: N/A\n"

        # Per-room status
        input_str += "--- Per-Room Status ---\n"
        for r in range(1, 8):
            room_key = f"room{r}"
            input_str += f"- Room {r}:\n"
            for k in room_input_keys:
                # Use robust .get() to handle cases where a key might not exist yet
                val = recorder[room_key].get(k, [None])[t]
                name = k.replace('_', ' ').title()

                if val is None:
                    input_str += f"  - {name}: N/A\n"
                elif k == "occupant_num":
                    # Occupant number is a dictionary, so we sum its values for the total
                    total_occupants = sum(val.values())
                    input_str += f"  - {name}: {total_occupants}\n"
                elif "temp" in k:
                    input_str += f"  - {name}: {val:.2f}°C\n"
                else:
                    input_str += f"  - {name}: {val}\n"

    except IndexError:
        return "Error: Could not generate detailed prompt due to recorder index out of bounds. Please provide action."

    # Combine with system prompt and instruction
    # Load system prompt
    with open("system_prompt.txt", "r") as f:
        system_prompt = f.read()
    instruction = "Provide the optimal FCU fan speed settings for each room based on the current room temperatures, FCU feedback, occupant numbers, and outdoor temperature."

    return f"{system_prompt}\n\n{instruction}{input_str}"

def call_llm_api(prompt: str) -> str:
    """Calls the local LLM API."""
    if client is None: return ""
    messages = [{"role": "user", "content": prompt}]
    try:
        result = client.chat.completions.create(messages=messages, model="test")
        return result.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        return ""

def parse_llm_response_to_array(response_text: str) -> list[int]:
    """
    Parses the natural language response from an LLM to extract FCU fan speed
    setpoints and returns them as a list of integers.

    This function uses regular expressions to find all occurrences of the fan speed
    setting command.

    Args:
        response_text: The string output from the language model.

    Returns:
        A list of integers representing the fan speed control values for each room.
        Returns an empty list if no matches are found.
    """
    # The regex pattern looks for the literal string "Set FCU Fan Speed: "
    # and then captures one or more digits (\d+) that follow.
    # The `re.findall` function returns a list of all captured groups.
    pattern = r"Set FCU Fan Speed: (\d+)"

    # Find all matches of the pattern in the input text
    matches = re.findall(pattern, response_text)

    # Convert the list of string matches (e.g., ['2', '2', '1']) to a list of integers
    control_array = [int(match) for match in matches]

    return control_array

# ==============================================================================
# Main Test Logic
# ==============================================================================

save_folder = "figure/0711_LLM_Agent_Test/"
test_name = "LLM_Agent_RecorderBasedPrompt"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

print(f"Starting test with LLM Agent: {test_name}")

reward_mode = "Baseline_OCC_PPD_with_energy"
tradeoff_constant = 10.0
frame_skip = 5

env1 = gym.make("SemiPhysBuildingSim-v0",
                reward_mode=reward_mode,
                tradeoff_constant=tradeoff_constant,
                eval_mode=True)
env1 = FrameSkip(env1, skip=frame_skip)

action_list = []
obs = env1.reset() # reset() populates the initial state in data_recorder
rewards = 0
done = False
i = 0
while not done:
    i += 1

    # *** CORE CHANGE HERE ***
    # Generate prompt from the recorder at the current step `i`
    prompt = state2language(env1, i-1)

    if i == 1:
        print("--- Example of Generated Prompt (Step 1) ---")
        print(prompt)
        print("------------------------------------------")

    llm_response = call_llm_api(prompt)
    action = parse_llm_response_to_array(llm_response)

    print(f"Step {i}: LLM Response -> '{llm_response[:80]}...', Parsed Action -> {action}")

    action_list.append(action)
    # The step function updates the recorder with new data for the *next* iteration
    obs, r, done, info = env1.step(action)
    rewards += r
print("Total rewards:" + str(rewards))


# ==============================================================================
# Plotting (No changes)
# ==============================================================================
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle("Agent: " + test_name, fontsize=16)
axes = axes.flatten()
data_recorder = env1.data_recorder
outdoor_temp = data_recorder["sensor_outdoor"]["outdoor_temp"]

for i_ax in range(7):
    ax = axes[i_ax]
    room_str = "room" + str(i_ax+1)
    room_temp = data_recorder[room_str]["room_temp"]
    occupancy = data_recorder[room_str]["occupant_num"]
    occupancy_sitting = [ o["sitting"] for o in occupancy]
    occupancy_standing = [ o["standing"] for o in occupancy]
    occupancy_walking = [ o["walking"] for o in occupancy]
    occupancy_total = [ sum(o.values()) for o in occupancy]

    ax.plot(room_temp, marker='o', linestyle='-', color='b', label='Temperature')
    ax.plot(outdoor_temp, marker='o', linestyle='-', color='r', label='Outdoor Temperature')
    ax.set_xlabel('Time Steps'); ax.set_ylabel('Value'); ax.set_title(room_str)
    ax.set_ylim(19, 31); ax.set_xlim(-20, 620)
    ax.yaxis.set_ticks(range(10, 30, 1)); ax.xaxis.set_ticks(range(0, 600, 60))
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    ax_twin = ax.twinx()
    ax_twin.plot(occupancy_sitting,  linestyle='-', color='m', label='sitting', alpha=0.7)
    ax_twin.plot(occupancy_standing,  linestyle='-', color='c', label='standing', alpha=0.7)
    ax_twin.plot(occupancy_walking,  linestyle='-', color='y', label='walking', alpha=0.7)
    ax_twin.plot(occupancy_total,  linestyle='-', color='k', label='total', alpha=1.0)
    ax_twin.set_ylabel('Occupancy (People)'); ax_twin.set_ylim(0, 11)
    ax_twin.yaxis.set_ticks(range(0, 5, 1))

    if i_ax == 6:
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

# Plotting summary statistics
ax2 = axes[8]; reward = data_recorder["training"]["reward"]
ax2.plot(reward, marker='o', linestyle='-', color='g', label='Reward')
ax2.set_title(f'Total Reward: {np.sum(reward):.1f}'); ax2.legend(); ax2.grid(True)

ax3 = axes[9]; FCU_power = data_recorder["training"]["energy_consumption"]
ax3.plot(FCU_power, marker='o', linestyle='-', color='g', label='FCU Power')
ax3.set_title(f'Total FCU Power: {np.sum(FCU_power):.1f}'); ax3.legend(); ax3.grid(True)

ax4 = axes[10]; pmv_mean = data_recorder["training"]["mean_pmv"]
ax4.plot(pmv_mean, marker='o', linestyle='-', color='g', label='PMV Mean')
ax4.set_title(f'Avg PMV Mean: {np.mean(pmv_mean):.2f}'); ax4.legend(); ax4.grid(True)

ax5 = axes[11]; ppd_mean = data_recorder["training"]["mean_ppd"]
ax5.plot(ppd_mean, marker='o', linestyle='-', color='g', label='PPD Mean')
ax5.set_title(f'Avg PPD Mean: {np.mean(ppd_mean):.2f}'); ax5.legend(); ax5.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(save_folder, test_name + '.png'))
plt.show()

env1.close()
import gc
gc.collect()
