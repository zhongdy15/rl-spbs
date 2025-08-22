import os
import pandas as pd
import json
import numpy as np

# --- Configuration ---
BASE_DATA_DIR = '../../expert_data/expert_data_remap'
OUTPUT_DIR = 'all_types_sft_data'
ROOM_IDS = list(range(1, 8))
DAYS = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]
YEAR = 2021
MONTH = 8

# Comfort settings for generating reasons
COMFORT_TEMP_MIN = 22.0
COMFORT_TEMP_MAX = 26.0

# --- Predefined System Prompt (from predefined_prompt.txt) ---
with open("predefined_prompt/instruction_prompt.txt", "r") as f:
    system_prompt = f.read()
PREDEFINED_SYSTEM_PROMPT = system_prompt

# --- SFT Prompt Templates ---
# INSTRUCTION_TEMPLATE 简化，因为系统角色和场景已由 PREDEFINED_SYSTEM_PROMPT 定义
INSTRUCTION_TEMPLATE = (
    "Analyze the real-time status of 7 rooms and identify which rooms most urgently require an adjustment to their Fan Coil Unit (FCU) gear settings. "
    "The goal is to maintain thermal comfort (temperature between 22°C and 26°C) while minimizing energy consumption. "
    "Based on the provided room states, output a prioritized list of rooms that need changes. If no changes are needed, provide an empty list. "
    "Your output must be a JSON object containing a 'rooms' list (of room IDs) and a 'reasons' list (explaining the prioritization)."
)


def format_state_as_input(state_data: dict) -> str:
    """Formats the combined state of all rooms into a single string for the SFT input."""
    input_parts = ["Current multi-room status:"]
    for room_id in sorted(state_data.keys()):
        state = state_data[room_id]
        # Use mean temperature and humidity for simplicity
        avg_temp = (state.get('room_temp1', 0) + state.get('room_temp2', 0)) / 2
        avg_rh = (state.get('room_RH1', 0) + state.get('room_RH2', 0)) / 2

        # print(state) # 这一行可能用于调试，在生产代码中通常移除
        input_parts.append(
            f"Room {room_id} - Temp: {avg_temp:.2f}°C, Humidity: {avg_rh:.2f}%, "
            f"Occupants: {int(state.get('occupant_num', 0))}, "
            f"Current Gear: {int(state.get('FCU_combined_state', 0))};"
        )

    return " ".join(input_parts)


def get_priority_and_reasons(changed_room_ids: list, current_states: dict) -> tuple:
    """
    Prioritizes rooms based on temperature deviation and generates reasons.
    """
    room_priorities = []
    for room_id in changed_room_ids:
        state = current_states[room_id]
        avg_temp = (state.get('room_temp1', 0) + state.get('room_temp2', 0)) / 2

        deviation = 0
        if avg_temp > COMFORT_TEMP_MAX:
            deviation = avg_temp - COMFORT_TEMP_MAX
        elif avg_temp < COMFORT_TEMP_MIN:
            deviation = COMFORT_TEMP_MIN - avg_temp

        room_priorities.append({'id': room_id, 'deviation': deviation, 'temp': avg_temp})

    # Sort rooms by deviation, descending
    sorted_rooms = sorted(room_priorities, key=lambda x: x['deviation'], reverse=True)

    # Generate output lists
    final_room_list = [r['id'] for r in sorted_rooms]
    reasons = []
    for r in sorted_rooms:
        if r['temp'] > COMFORT_TEMP_MAX:
            reasons.append(
                f"Room {r['id']}'s temperature ({r['temp']:.1f}°C) is above the comfort range (> {COMFORT_TEMP_MAX}°C).")
        elif r['temp'] < COMFORT_TEMP_MIN:
            reasons.append(
                f"Room {r['id']}'s temperature ({r['temp']:.1f}°C) is below the comfort range (< {COMFORT_TEMP_MIN}°C).")
        else:
            reasons.append(f"Room {r['id']}'s gear was adjusted to maintain comfort or respond to occupancy changes.")

    return final_room_list, reasons


def create_sft_sample(input_str: str, output_obj: dict) -> dict:
    """Creates a single SFT sample in the required format."""
    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": input_str,
        "output": json.dumps(output_obj, indent=2),  # Output is a JSON string
        "system": PREDEFINED_SYSTEM_PROMPT, # 从这里融入 predefined_prompt.txt 的内容
        "history": []
    }


def process_data():
    """Main function to process all CSVs and generate SFT datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sft_data_type1 = []  # Stable Off
    sft_data_type2 = []  # Stable On
    sft_data_type3 = []  # Action Taken

    for day in DAYS:
        print(f"--- Processing data for 2021-08-{day} ---")

        # 1. Aggregate data for all rooms for the current day
        daily_dfs = []
        is_day_complete = True
        for room_id in ROOM_IDS:
            file_name = f'room_{room_id}_{YEAR}_{MONTH}_{day}_0900_2100.csv'
            file_path = os.path.join(BASE_DATA_DIR, file_name)

            if not os.path.exists(file_path):
                print(f"Warning: File not found for Room {room_id} on day {day}. Skipping day.")
                is_day_complete = False
                break

            df = pd.read_csv(file_path)
            # Create a datetime index for merging
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y/%m/%d %H:%M')
            df = df.set_index('datetime')
            # Add room_id prefix to columns to avoid clashes
            df = df.add_prefix(f'room_{room_id}_')
            daily_dfs.append(df)

        if not is_day_complete:
            continue

        # Merge all room dataframes for the day
        combined_df = pd.concat(daily_dfs, axis=1)

        # 2. Iterate through timestamps and generate SFT samples
        for i in range(1, len(combined_df)):
            current_row = combined_df.iloc[i]
            prev_row = combined_df.iloc[i - 1]

            # Prepare state data for input formatting and logic
            current_states = {}
            prev_states_fcu = {}
            current_states_fcu = {}

            for room_id in ROOM_IDS:
                current_states[room_id] = {col.split('_', 2)[-1]: current_row[col]
                                           for col in combined_df.columns if col.startswith(f'room_{room_id}_')}
                prev_states_fcu[room_id] = prev_row[f'room_{room_id}_FCU_combined_state']
                current_states_fcu[room_id] = current_row[f'room_{room_id}_FCU_combined_state']

            # Identify rooms where the gear changed
            changed_room_ids = [
                room_id for room_id in ROOM_IDS
                if prev_states_fcu[room_id] != current_states_fcu[room_id]
            ]

            # Format the input string (common for all types)
            sft_input_str = format_state_as_input(current_states)

            # 3. Categorize and create SFT sample
            if not changed_room_ids:
                # No rooms changed state
                is_all_off = all(state == 0 for state in current_states_fcu.values())

                if is_all_off:
                    # Type 1: Stable Off
                    output_obj = {
                        "rooms": [],
                        "reasons": ["All rooms are stable and off. To save energy, no adjustments are needed."]
                    }
                    sft_data_type1.append(create_sft_sample(sft_input_str, output_obj))
                else:
                    # Type 2: Stable On
                    output_obj = {
                        "rooms": [],
                        "reasons": [
                            "The current fan settings are effectively maintaining conditions. Continue monitoring and no adjustments are needed now."]
                    }
                    sft_data_type2.append(create_sft_sample(sft_input_str, output_obj))
            else:
                # Type 3: Action Taken
                prio_rooms, reasons = get_priority_and_reasons(changed_room_ids, current_states)
                output_obj = {"rooms": prio_rooms, "reasons": reasons}
                sft_data_type3.append(create_sft_sample(sft_input_str, output_obj))

    # 4. Save the generated data to JSON files
    with open(os.path.join(OUTPUT_DIR, 'type1_stable_off.json'), 'w', encoding='utf-8') as f:
        json.dump(sft_data_type1, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, 'type2_stable_on.json'), 'w', encoding='utf-8') as f:
        json.dump(sft_data_type2, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, 'type3_action_taken.json'), 'w', encoding='utf-8') as f:
        json.dump(sft_data_type3, f, indent=2, ensure_ascii=False)

    print("\n--- SFT Dataset Generation Complete ---")
    print(f"Generated {len(sft_data_type1)} samples for Type 1 (Stable Off).")
    print(f"Generated {len(sft_data_type2)} samples for Type 2 (Stable On).")
    print(f"Generated {len(sft_data_type3)} samples for Type 3 (Action Taken).")
    print(f"All files saved in the '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    process_data()
