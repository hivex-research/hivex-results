import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import math
import pandas as pd
import matplotlib.colors as mcolors
import re


def get_difficulty_or_pattern(name):
    match = re.search(r"(pattern|difficulty)_(\d+)", name)
    if match:
        return match.group(0), int(
            match.group(2)
        )  # Change group(1) to group(2) for the integer
    else:
        return ("None", "None")


def get_agent_count(name):
    match = re.search(r"(agent_count)_(\d+)", name)
    if match:
        return match.group(0), int(
            match.group(2)
        )  # Change group(1) to group(2) for the integer
    else:
        return ("None", "None")


def get_task(name) -> int:
    match = re.search(r"task_(\d+)", name)
    if match:
        return match.group(0), int(match.group(1))
    else:
        return ("None", "None")


def get_id(name) -> int:
    match = re.search(r"id_(\d+)", name)
    if match:
        return match.group(0), int(match.group(1))
    else:
        return ("None", "None")


def extract_data_from_event_file(file_path):
    data = []
    for summary in tf.compat.v1.train.summary_iterator(str(file_path)):
        for value in summary.summary.value:
            if value.HasField("simple_value"):
                data.append(
                    {
                        "step": summary.step,
                        "tag": value.tag,
                        "value": value.simple_value,
                    }
                )
    return data


def list_directories(root_dir, label):
    all_data = []
    root_path = Path(root_dir)
    for path in root_path.rglob("*"):
        if (
            path.is_dir()
            and str(path).endswith("Agent")
            and "agent_count" in path.parent.name
        ):
            name = path.parent.name
            env_name = name.split("_")[0]
            difficulty_or_pattern = get_difficulty_or_pattern(name)
            difficulty_or_pattern_key = difficulty_or_pattern[0].split("_")[0]
            difficulty_or_pattern_value = difficulty_or_pattern[1]
            agent_count = get_agent_count(name)[1]
            task = get_task(name)[1]
            id = get_id(name)[1]
            print(f"Processing: {name}")
            print(
                f"{difficulty_or_pattern_key}: {difficulty_or_pattern_value}, Task: {task}, ID: {id}, Agent Count: {agent_count}"
            )

            # Search for the specific file inside the "Agent" directory
            for file in path.iterdir():
                if file.name.startswith("events.out.tfevents"):
                    # Extract data from the event file
                    data = extract_data_from_event_file(file)
                    # Append additional context data
                    for entry in data:
                        entry["difficulty_or_pattern_key"] = difficulty_or_pattern_key
                        entry["difficulty_or_pattern_value"] = (
                            difficulty_or_pattern_value
                        )
                        entry["task"] = task
                        entry["agent_count"] = agent_count
                        entry["id"] = id
                        entry["env_name"] = env_name
                        entry["source"] = label
                    all_data.extend(data)

    return pd.DataFrame(all_data)


import matplotlib.pyplot as plt
import pandas as pd


def plot_average_reward_for_each_agent_count(data_list, output_path):
    tag_list = [
        ["Environment/Cumulative Reward"],
        [
            "Environment/Cumulative Reward",
            "DroneBasedReforestation/Tree Drop Count",
            "DroneBasedReforestation/Recharge Energy Count",
        ],
        [
            "Environment/Cumulative Reward",
            "AerialWildfireSuppression/Extinguishing Trees Reward",
        ],
    ]

    # Define the colors
    colors = ["#14DFB4", "#FF931E", "#FF1D25"]

    # Custom titles for each plot
    titles = [
        "Wind Farm Control",
        "Drone-Based Reforestation",
        "Aerial Wildfire Suppression",
    ]

    # Create subplots for each dataset in data_list with smaller scale
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # Adjusted figure size

    # Loop through each dataset and corresponding tags_of_interest
    for i, (data, tags_of_interest) in enumerate(zip(data_list, tag_list)):

        # Convert the data into a DataFrame
        df = pd.DataFrame(data)

        # Filter the dataframe for the tags of interest
        df_filtered_extended = df[df["tag"].isin(tags_of_interest)]

        # Group by 'tag' and 'agent_count', then calculate the mean value
        df_grouped = df_filtered_extended.groupby(
            ["tag", "agent_count"], as_index=False
        )["value"].mean()

        # Plotting each tag's average value in the corresponding subplot
        for tag, color in zip(tags_of_interest, colors):
            df_tag_grouped = df_grouped[df_grouped["tag"] == tag]
            axes[i].plot(
                df_tag_grouped["agent_count"],
                df_tag_grouped["value"],
                marker="o",
                label=tag,
                color=color,
            )

        # Labeling for each subplot
        axes[i].set_xlabel("Agent Count", fontsize=12)
        axes[i].set_title(titles[i], fontsize=14)  # Set the custom title

        # Add grid to each subplot
        axes[i].grid(True)

        # Add legend to each subplot
        axes[i].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize=10
        )

    # Set the common y-axis label on the left side
    fig.text(0.05, 0.67, "Average Value", va="center", rotation="vertical", fontsize=14)

    # Adjust layout to fit everything cleanly
    plt.tight_layout(
        rect=[0.05, 0.05, 1, 0.95]
    )  # Adjust layout to leave space for labels

    # Reduce white space and save the figure as a PDF
    fig.savefig(output_path, format="pdf", bbox_inches="tight")

    # Show the plots
    plt.show()


path_prefix = "C:/Users/pdsie/Documents/hivex-results/results/"

test_dirs = [
    f"{path_prefix}WindFarmControl/test",
    f"{path_prefix}DroneBasedReforestation/test",
    f"{path_prefix}AerialWildfireSuppression/test",
]
output_path = f"{path_prefix}scalability_agent_count.pdf"

datasets = []

for test_root_dir in test_dirs:
    test_data = list_directories(test_root_dir, "test")
    datasets.append(test_data)

plot_average_reward_for_each_agent_count(datasets, output_path)
