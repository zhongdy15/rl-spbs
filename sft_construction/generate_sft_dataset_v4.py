import pandas as pd
import json
import os
from datetime import datetime

# --- Constants ---
with open("predefined_prompt/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open("predefined_prompt/instruction_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTION_PROMPT = f.read()

OPTIMAL_TEMP_MIN = 25
OPTIMAL_TEMP_MAX = 27

DATA_DIR = "../expert_data/expert_data_remap"
OUTPUT_DIR = "all_types_sft_data"

ROOM_IDS = range(1, 8)
DAYS = [7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21]  # 8月份的特定天

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and Aggregate Data ---
room_dfs = {}

print("Loading and preprocessing data for each room...")
for room_id in ROOM_IDS:
    room_data_frames = []
    for day in DAYS:
        # Construct file path based on provided pattern
        file_path = os.path.join(DATA_DIR, f"room_{room_id}_2021_8_{day}_0900_2100.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            room_data_frames.append(df)
        else:
            pass

    if room_data_frames:
        combined_df = pd.concat(room_data_frames, ignore_index=True)
        # Convert date and time columns to a single datetime object
        combined_df['datetime'] = pd.to_datetime(combined_df['date'] + ' ' + combined_df['time'],
                                                 format='%Y/%m/%d %H:%M')
        # Sort by datetime to ensure correct 'shift' for previous state
        combined_df = combined_df.sort_values(by='datetime').reset_index(drop=True)

        # Calculate previous FCU_combined_state for categorization and output logic
        combined_df['prev_FCU_combined_state'] = combined_df['FCU_combined_state'].shift(1)
        room_dfs[room_id] = combined_df
    else:
        print(f"No data loaded for room {room_id}. This room will be excluded from common timestamps consideration.")

# Find common timestamps across all rooms
print("Finding common timestamps across all rooms...")
if not room_dfs:
    print("No room data loaded. Exiting.")
    exit()

first_room_id_with_data = next((rid for rid in ROOM_IDS if rid in room_dfs), None)
if first_room_id_with_data is None:
    print("No data available for any room. Exiting.")
    exit()

common_timestamps_set = set(room_dfs[first_room_id_with_data]['datetime'])

for room_id in ROOM_IDS:
    if room_id in room_dfs:
        common_timestamps_set = common_timestamps_set.intersection(set(room_dfs[room_id]['datetime']))
    else:
        pass

common_timestamps = sorted(list(common_timestamps_set))

print(f"Found {len(common_timestamps)} common timestamps where all 7 rooms have data.")

# Initialize lists for SFT data categories
sft_data_type1 = []  # All rooms: prev FCU == 0 and curr FCU == 0
sft_data_type2 = []  # All rooms: prev FCU == curr FCU but not all are 0; OR mixed stable
sft_data_type3 = []  # At least one room: prev FCU != curr FCU

# --- Generate SFT Data Entries ---
print("Generating SFT data entries...")
if len(common_timestamps) > 1:
    for i in range(1, len(common_timestamps)):
        timestamp = common_timestamps[i]

        input_lines = []
        current_room_states = {}
        previous_fcu_states = {}
        current_fcu_states = {}

        is_valid_entry = True
        current_outdoor_temp = None  # Initialize outdoor temp for this timestamp

        # Get outdoor temperature once for the current timestamp
        # Assuming 'outdoor_temp' is present in each room's CSV for the given timestamp
        outdoor_temp_data_row = room_dfs[ROOM_IDS[0]].loc[room_dfs[ROOM_IDS[0]]['datetime'] == timestamp]
        if not outdoor_temp_data_row.empty:
            current_outdoor_temp = outdoor_temp_data_row['outdoor_temp'].iloc[0]
        else:
            is_valid_entry = False  # If outdoor temp data is missing from first room, skip

        if not is_valid_entry:
            continue  # Skip this timestamp if outdoor temp data is missing

        for room_id in ROOM_IDS:
            room_data_at_ts = room_dfs[room_id].loc[room_dfs[room_id]['datetime'] == timestamp]

            if room_data_at_ts.empty:
                is_valid_entry = False
                break

                # Changed: Calculate mean of room_temp1 and room_temp2
            curr_temp = (room_data_at_ts['room_temp1'].iloc[0] + room_data_at_ts['room_temp2'].iloc[0]) / 2
            curr_occupant = room_data_at_ts['occupant_num'].iloc[0]
            curr_fan_speed = room_data_at_ts['FCU_combined_state'].iloc[0]
            prev_fan_speed = room_data_at_ts['prev_FCU_combined_state'].iloc[0]

            if pd.isna(prev_fan_speed):
                is_valid_entry = False
                break

            input_lines.append(
                f"Room {room_id} — Temperature: {curr_temp:.2f}°C, Occupant Num: {int(curr_occupant)}, Fan Speed: {int(curr_fan_speed)};")

            current_room_states[room_id] = {
                'temp': curr_temp,
                'occupant': curr_occupant,
                'fan_speed': curr_fan_speed
            }
            previous_fcu_states[room_id] = prev_fan_speed
            current_fcu_states[room_id] = curr_fan_speed

        if not is_valid_entry:
            continue

            # Add outdoor temperature line to input_lines before joining
        input_lines.append(f"Outdoor - Temperature: {current_outdoor_temp:.2f}°C")

        explanation_line = "Here is the current room temperature, occupancy, and fan speed status:"
        input_str = explanation_line + "\n" + "\n".join(input_lines)

        # --- Determine Output JSON based on FCU changes and current conditions ---
        output_rooms = []
        output_reasons = []
        has_any_actual_fcu_change = False

        for room_id in ROOM_IDS:
            curr_fcu = current_fcu_states[room_id]
            prev_fcu = previous_fcu_states[room_id]
            temp = current_room_states[room_id]['temp']
            occupant = current_room_states[room_id]['occupant']

            if curr_fcu != prev_fcu:
                has_any_actual_fcu_change = True
                output_rooms.append(room_id)

                # Formulate reason for the actual change, handling occupant_num = -1 and outdoor temp
                occupant_desc = ""
                if int(occupant) == -1:
                    occupant_desc = " (Occupant data unavailable (sensor error))"
                elif int(occupant) > 0:
                    occupant_desc = f" ({int(occupant)} occupants)"

                outdoor_desc = ""
                # Add outdoor temperature context if it's relevant (e.g., high)
                if current_outdoor_temp > OPTIMAL_TEMP_MAX + 1:  # Heuristic: if outdoor is significantly above comfort range
                    outdoor_desc = f" (Outdoor temp: {current_outdoor_temp:.2f}°C)"

                reason_prefix = f"Room {room_id}'s temperature ({temp:.2f}°C)"
                reason_text = ""

                if temp > OPTIMAL_TEMP_MAX:
                    reason_text = f"{reason_prefix} exceeds the comfort range (> {OPTIMAL_TEMP_MAX}°C){occupant_desc}{outdoor_desc}."
                elif temp < OPTIMAL_TEMP_MIN:
                    # Outdoor temp is less relevant for "too cold" inside, unless fan was on.
                    if curr_fcu > 0 or (curr_fcu != prev_fcu):  # If fan active or was adjusted down/off
                        reason_text = f"{reason_prefix} is below the comfort range (< {OPTIMAL_TEMP_MIN}°C) while the fan is active at speed {int(curr_fcu)}{occupant_desc}. Consider reducing fan speed or turning it off."
                    else:  # Fan was already off and it's cold
                        reason_text = f"{reason_prefix} is below the comfort range (< {OPTIMAL_TEMP_MIN}°C){occupant_desc}."
                elif OPTIMAL_TEMP_MIN <= temp <= OPTIMAL_TEMP_MAX:
                    # If temperature is optimal but FCU changed
                    if curr_fcu == 0 and prev_fcu > 0:  # Fan turned off when comfortable
                        reason_text = f"{reason_prefix} is within the comfort range. The fan was turned off (from speed {int(prev_fcu)}) for energy efficiency or to prevent overcooling{occupant_desc}."
                    elif curr_fcu > 0 and prev_fcu == 0:  # Fan turned on when comfortable (e.g., anticipating load)
                        reason_text = f"{reason_prefix} is within the comfort range, but the fan was turned on to {int(curr_fcu)} (from off), possibly in anticipation of higher load or to maintain stricter control{outdoor_desc}{occupant_desc}."
                    else:  # Fan speed changed between non-zero values when comfortable
                        if curr_fcu > prev_fcu:  # Speed increased
                            reason_text = f"{reason_prefix} is within the comfort range. The fan speed was increased from {int(prev_fcu)} to {int(curr_fcu)} for fine-tuning or to better cope with external conditions{outdoor_desc}{occupant_desc}."
                        else:  # Speed decreased
                            reason_text = f"{reason_prefix} is within the comfort range. The fan speed was decreased from {int(prev_fcu)} to {int(curr_fcu)} for energy efficiency or as the room's cooling needs reduced{occupant_desc}."

                if not reason_text:  # Fallback for any unhandled edge cases
                    reason_text = f"Room {room_id}'s fan speed changed from {int(prev_fcu)} to {int(curr_fcu)}{occupant_desc}{outdoor_desc}."

                output_reasons.append(reason_text)

        output_json = {}
        if has_any_actual_fcu_change:
            output_json = {
                "rooms": sorted(list(set(output_rooms))),
                "reasons": output_reasons
            }
        else:
            all_fans_off_now = True
            for room_id in ROOM_IDS:
                if current_room_states[room_id]['fan_speed'] > 0:
                    all_fans_off_now = False
                    break

            if all_fans_off_now:
                output_json = {
                    "rooms": [],
                    "reasons": ["All rooms are stable and no adjustments are needed at this moment."]
                }
            else:
                output_json = {
                    "rooms": [],
                    "reasons": [
                        "The current fan settings are maintaining desired room conditions. No adjustments are needed now."]
                }

        output_str = json.dumps(output_json, indent=2, ensure_ascii=False)

        is_type3_category = False
        for room_id in ROOM_IDS:
            if current_fcu_states[room_id] != previous_fcu_states[room_id]:
                is_type3_category = True
                break

        is_type1_category = True
        if not is_type3_category:
            for room_id in ROOM_IDS:
                if not (current_fcu_states[room_id] == 0 and previous_fcu_states[room_id] == 0):
                    is_type1_category = False
                    break
        else:
            is_type1_category = False

        sft_entry = [{
            "instruction": INSTRUCTION_PROMPT,
            "input": input_str,
            "output": output_str,
            "system": SYSTEM_PROMPT,
            "history": []
        }]

        if is_type3_category:
            sft_data_type3.append(sft_entry)
        elif is_type1_category:
            sft_data_type1.append(sft_entry)
        else:
            sft_data_type2.append(sft_entry)

else:
    print("Not enough common timestamps to generate SFT data (need at least two for comparison).")

print(f"Generated {len(sft_data_type1)} Type 1 entries.")
print(f"Generated {len(sft_data_type2)} Type 2 entries.")
print(f"Generated {len(sft_data_type3)} Type 3 entries.")

# --- Save SFT Data ---
print("Saving SFT data to JSON files...")

# Unpack the single-element lists before dumping
with open(os.path.join(OUTPUT_DIR, "type1_stable_off.json"), "w", encoding="utf-8") as f:
    json.dump([item[0] for item in sft_data_type1], f, indent=2, ensure_ascii=False)
print(f"Saved Type 1 data to {os.path.join(OUTPUT_DIR, 'type1_stable_off.json')}")

with open(os.path.join(OUTPUT_DIR, "type2_stable_on.json"), "w", encoding="utf-8") as f:
    json.dump([item[0] for item in sft_data_type2], f, indent=2, ensure_ascii=False)
print(f"Saved Type 2 data to {os.path.join(OUTPUT_DIR, 'type2_stable_on.json')}")

with open(os.path.join(OUTPUT_DIR, "type3_adjusted.json"), "w", encoding="utf-8") as f:
    json.dump([item[0] for item in sft_data_type3], f, indent=2, ensure_ascii=False)
print(f"Saved Type 3 data to {os.path.join(OUTPUT_DIR, 'type3_adjusted.json')}")

print("SFT dataset construction complete.")
