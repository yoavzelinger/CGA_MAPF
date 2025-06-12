import os
import pandas as pd

SCENARIOS_FOLDER_PATH = 'scenarios'
RAW_SCENARIOS_FOLDER_PATH = os.path.join(SCENARIOS_FOLDER_PATH, 'raw')

def extract_unique_agent_goal_pairs():
    for file_index, filename in enumerate(sorted(os.listdir(RAW_SCENARIOS_FOLDER_PATH))):
        if filename.endswith(".scen"):
            scenario_path = os.path.join(RAW_SCENARIOS_FOLDER_PATH, filename)
            with open(scenario_path, 'r') as file:
                lines = file.readlines()

            data = [line.strip().split() for line in lines[1:]]
            original_length = len(data)
            df = pd.DataFrame(data, columns=[
                'bucket', 'map', 'map_width', 'map_height',
                'start_x', 'start_y', 'goal_x', 'goal_y', 'distance'
            ])

            df = df.astype({'start_x': int, 'start_y': int, 'goal_x': int, 'goal_y': int})

            seen_starts = set()
            seen_goals = set()
            unique_rows = []

            map_name = None
            for _, row in df.iterrows():
                if map_name is None:
                    map_name = os.path.splitext(row['map'])[0]

                start = (row['start_x'], row['start_y'])
                goal = (row['goal_x'], row['goal_y'])

                if start in seen_starts or goal in seen_goals: # avoid duplicate start / goal pairs
                    continue

                seen_starts.add(start)
                seen_goals.add(goal)
                unique_rows.append(row)


            unique_df = pd.DataFrame(unique_rows)
            output_df = unique_df[['start_x', 'start_y', 'goal_x', 'goal_y']]

            output_filename = f"{map_name}__scenario_{(file_index % 25) + 1}.csv"
            output_path = os.path.join(SCENARIOS_FOLDER_PATH, output_filename)
            output_df.to_csv(output_path, index=False)

            print(f"file {output_filename} created with unique agent-goal pairs from {filename}")
            print(f"\tOriginal length: {original_length}, Unique length: {len(unique_rows)}")
            print()

if __name__ == "__main__":
    extract_unique_agent_goal_pairs()
    print(f"All files processed.")