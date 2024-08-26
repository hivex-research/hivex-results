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
        if path.is_dir() and str(path).endswith("Agent"):
            name = path.parent.name
            env_name = name.split("_")[0]
            difficulty_or_pattern = get_difficulty_or_pattern(name)
            difficulty_or_pattern_key = difficulty_or_pattern[0].split("_")[0]
            difficulty_or_pattern_value = difficulty_or_pattern[1]
            task = get_task(name)[1]
            id = get_id(name)[1]
            print(f"Processing: {name}")
            print(
                f"{difficulty_or_pattern_key}: {difficulty_or_pattern_value}, Task: {task}, ID: {id}"
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
                        entry["id"] = id
                        entry["env_name"] = env_name
                        entry["source"] = label
                    all_data.extend(data)

    return pd.DataFrame(all_data)


# Function to downsample data
def downsample(df, factor=4):
    return df.iloc[::factor, :]


def plot_data_for_groups(combined_data_frame, output_path, down_sample_factor=4):
    # Check if the required columns exist
    required_columns = {
        "step",
        "tag",
        "value",
        "difficulty_or_pattern_key",
        "difficulty_or_pattern_value",
        "task",
        "env_name",
        "source",
    }
    if not required_columns.issubset(combined_data_frame.columns):
        print(
            f"DataFrame is missing required columns. Found columns: {combined_data_frame.columns}"
        )
        return

    # Group by pattern, task, and environment name
    grouped = combined_data_frame.groupby(
        ["difficulty_or_pattern_key", "difficulty_or_pattern_value", "task", "env_name"]
    )

    for (
        difficulty_or_pattern_key,
        difficulty_or_pattern_value,
        task,
        env_name,
    ), group in grouped:

        # Get unique tags
        unique_tags = group["tag"].unique()

        excluded_tags = [
            "Environment/Lesson Number/pattern",
            "Environment/Lesson Number/difficulty",
            "Environment/Lesson Number/task",
        ]

        # Remove excluded tags from unique tags
        filtered_tags = [tag for tag in unique_tags if tag not in excluded_tags]

        # Determine the number of rows and columns for the grid
        num_tags = len(filtered_tags)
        num_cols = 6  # Number of columns for A4 ratio
        num_rows = math.ceil(num_tags / num_cols)

        # Set up the figure with an A4 ratio (1.414 height/width)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(11.69 * 2, 11.69 * 0.17 * num_rows)
        )  #  * 1.414))

        difficulty_or_pattern_label = (
            f"{difficulty_or_pattern_key.capitalize()}: {difficulty_or_pattern_value}, "
            if difficulty_or_pattern_key != "None"
            else ""
        )

        fig.suptitle(
            f"{env_name} Train & Test Metrics: {difficulty_or_pattern_label}Task: {task}",
            fontsize=16,
        )

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Define custom colors for training and testing
        palette = {"training": "#FF1D25", "test": "#14DFB4"}

        # Plot each tag in its own subplot
        for i, tag in enumerate(filtered_tags):
            ax = axes[i]
            data = group[group["tag"] == tag]
            if down_sample_factor > 1:
                data = downsample(data, down_sample_factor)
            sns.lineplot(
                data=data,
                x="step",
                y="value",
                hue="source",
                palette=palette,
                ax=ax,
            )
            ax.set_title(tag.split("/")[1])
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        # Save the figure as an SVG file with a more specific name
        output_file = f"{output_path}/{env_name}_results_{difficulty_or_pattern_key}_{difficulty_or_pattern_value}_task_{task}.pdf"
        plt.savefig(output_file, format="pdf")
        plt.close()


def plot_aggregated_matrices_on_one_sheet(data, output_path):
    # Get unique tags, tasks, and patterns/difficulties
    unique_tags = data["tag"].unique()

    excluded_tags = [
        "Environment/Lesson Number/pattern",
        "Environment/Lesson Number/difficulty",
        "Environment/Lesson Number/task",
        "Losses/Policy Loss",
        "Policy/Learning Rate",
        "Policy/Beta",
        "Policy/Epsilon",
    ]

    # Remove excluded tags from unique tags
    filtered_tags = [tag for tag in unique_tags if tag not in excluded_tags]

    tasks = data["task"].unique()
    patterns = data["difficulty_or_pattern_value"].unique()

    # Calculate the number of subplots needed
    num_tags = len(filtered_tags)
    ncols = 2  # Number of columns in the subplot grid
    nrows = (num_tags + 1) // ncols  # Calculate the number of rows needed

    # Set up the figure size to keep cells roughly the same size
    cell_width = 8
    cell_height = 2
    fig_width = cell_width * ncols
    fig_height = cell_height * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    fig.suptitle(f"Average Values for All Tags: {data['env_name'][0]}", fontsize=16)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Define the custom color map
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_gradient", ["#14DFB4", "#FF931E", "#FF1D25"]
    )

    for i, tag in enumerate(filtered_tags):
        # Initialize a matrix to store the average values with NaNs
        avg_matrix = pd.DataFrame(index=tasks, columns=patterns, dtype=float)

        for task in tasks:
            for pattern in patterns:
                # Filter data for the current task, pattern/difficulty, and tag
                task_pattern_data = data[
                    (data["task"] == task)
                    & (data["difficulty_or_pattern_value"] == pattern)
                    & (data["tag"] == tag)
                ]

                # Calculate the average value and fill the matrix
                if not task_pattern_data.empty:
                    avg_value = task_pattern_data["value"].mean()
                    avg_matrix.loc[task, pattern] = round(avg_value, 3)

        # Format the matrix to show 0.0 for 0.000 values
        avg_matrix = avg_matrix.map(lambda x: 0.0 if x == 0 else x)

        formatted_annotations = avg_matrix.map(
            lambda x: (0.0 if x == 0 else (f"{x:.3f}" if x < 1 else f"{x:.1f}"))
        )

        # Plot the matrix
        sns.heatmap(
            avg_matrix,
            annot=formatted_annotations,
            cmap=custom_cmap,
            cbar=True,
            ax=axes[i],
            fmt="",
        )
        sub_title = (
            tag if tag.split("/")[0] not in data["env_name"][0] else tag.split("/")[1]
        )
        axes[i].set_title(f"Average Values for Tag: {sub_title}")
        axes[i].set_xlabel("Pattern/Difficulty")
        axes[i].set_ylabel("Task")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Save as SVG
    plt.savefig(
        f"{output_path}/{data['env_name'].iloc[0]}_average_sheet.pdf", format="pdf"
    )
    plt.close()


def plot_cumulative_reward_multiple(data_list, output_path):
    tag_list = [
        ["Environment/Cumulative Reward", "Losses/Policy Loss"],
        [
            "Environment/Cumulative Reward",
            "WildfireResourceManagement/Individual Performance",
        ],
        [
            "Environment/Cumulative Reward",
            "DroneBasedReforestation/Recharge Energy Count",
        ],
        ["Environment/Cumulative Reward", "OceanPlasticCollector/Local Reward"],
        [
            "Environment/Cumulative Reward",
            "AerialWildfireSuppression/Extinguishing Trees Reward",
        ],
    ]

    # Determine the grid size for subplots
    num_datasets = len(data_list)
    grid_cols = 2
    grid_rows = num_datasets

    cell_width = 1.5
    cell_height = 1

    fig_width = cell_width * grid_cols * 6
    fig_height = cell_height * grid_rows * 2

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_gradient", ["#14DFB4", "#FF931E", "#FF1D25"]
    )

    for idx, data in enumerate(data_list):
        for jdx, tag in enumerate(tag_list[idx]):
            tasks = data["task"].unique()
            patterns = data["difficulty_or_pattern_value"].unique()

            avg_matrix = pd.DataFrame(index=tasks, columns=patterns, dtype=float)

            for task in tasks:
                for pattern in patterns:
                    task_pattern_data = data[
                        (data["task"] == task)
                        & (data["difficulty_or_pattern_value"] == pattern)
                        & (data["tag"] == tag)
                    ]

                    if not task_pattern_data.empty:
                        avg_value = task_pattern_data["value"].mean()
                        avg_matrix.loc[task, pattern] = round(avg_value, 3)

            avg_matrix = avg_matrix.map(lambda x: 0.0 if x == 0 else x)
            formatted_annotations = avg_matrix.map(
                lambda x: (0.0 if x == 0 else (f"{x:.3f}" if x < 1 else f"{x:.1f}"))
            )

            ax = axes[idx * 2 + jdx]

            # Plotting heatmap with formatted annotations
            sns.heatmap(
                avg_matrix,
                annot=formatted_annotations,
                cmap=custom_cmap,
                cbar=True,
                ax=ax,
                fmt="",
            )

            if data["difficulty_or_pattern_key"][0] == "None":
                ax.set_xlabel("Difficulty")
            else:
                ax.set_xlabel(f"{data['difficulty_or_pattern_key'][0].capitalize()}")
            ax.set_ylabel("Task")
            ax.set_title(f"{data['env_name'].iloc[0]}: {tag.split('/')[1]}")

    # Remove any unused subplots
    for idx in range(len(data_list) * len(tag_list), len(axes)):
        print(f"removing: {idx}")
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(f"{output_path}/cumulative_reward_multiple.pdf", format="pdf")
    plt.close()


path_prefix = "C:/Users/pdsie/Documents/hivex-results/results/"

train_dirs = [
    f"{path_prefix}WindFarmControl/train",
    f"{path_prefix}WildfireResourceManagement/train",
    f"{path_prefix}DroneBasedReforestation/train",
    f"{path_prefix}OceanPlasticCollection/train",
    f"{path_prefix}AerialWildfireSuppression/train",
]
test_dirs = [
    f"{path_prefix}WindFarmControl/test",
    f"{path_prefix}WildfireResourceManagement/test",
    f"{path_prefix}DroneBasedReforestation/test",
    f"{path_prefix}OceanPlasticCollection/test",
    f"{path_prefix}AerialWildfireSuppression/test",
]
output_paths = [
    f"{path_prefix}WindFarmControl/",
    f"{path_prefix}WildfireResourceManagement/",
    f"{path_prefix}DroneBasedReforestation/",
    f"{path_prefix}OceanPlasticCollection/",
    f"{path_prefix}AerialWildfireSuppression/",
]

if __name__ == "__main__":

    datasets = []

    for train_root_dir, test_root_dir, output_path in zip(
        train_dirs, test_dirs, output_paths
    ):
        training_data = list_directories(
            train_root_dir,
            "training",
        )
        test_data = list_directories(test_root_dir, "test")

        datasets.append(test_data)

        #### 1

        plot_data_for_groups(pd.concat([training_data, test_data]), output_path, 1)

        #### 2

        plot_aggregated_matrices_on_one_sheet(test_data, output_path)

    ### 3

    plot_cumulative_reward_multiple(
        datasets, "C:/Users/pdsie/Documents/hivex-results/results"
    )
