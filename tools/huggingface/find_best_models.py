import tensorflow as tf
from pathlib import Path
import pandas as pd
import re
import json


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
        if (
            path.is_dir()
            and str(path).endswith("Agent")
            and "agent_count" not in path.parent.name
        ):
            name = path.parent.name
            env_name = name.split("_")[0]
            difficulty_or_pattern = get_difficulty_or_pattern(name)
            difficulty_or_pattern_key = difficulty_or_pattern[0].split("_")[0]
            difficulty_or_pattern_value = difficulty_or_pattern[1]
            task = get_task(name)[1]
            id = get_id(name)[1]
            # print(f"Processing: {name}")
            # print(
            #     f"{difficulty_or_pattern_key}: {difficulty_or_pattern_value}, Task: {task}, ID: {id}"
            # )

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


def find_best_models(
    datasets,
    tags_of_interest,
    difficulty_or_pattern_of_interest,
    output_json_path,
    tag_criteria,
    task_tag_criteria,
):

    # Initialize a final dictionary to accumulate all results across datasets
    final_dict = {}

    # Iterate through datasets, tags, and difficulty/patterns of interest
    for (dataset_name, data), tags, d_or_p_list in zip(
        list(datasets.items()), tags_of_interest, difficulty_or_pattern_of_interest
    ):
        if dataset_name == "OceanPlasticCollection":
            dataset_name = "OceanPlasticCollector"

        # Convert the data into a DataFrame
        df = pd.DataFrame(data)

        # Print the tags and difficulty/patterns being used for filtering
        # print(
        #     f"Processing dataset: {dataset_name}, tags: {tags} and difficulty/pattern values: {d_or_p_list}"
        # )

        # Initialize a dictionary for the current dataset
        grouped_dict = {}

        # Check if d_or_p_list is empty
        if not d_or_p_list:
            # If the list is empty, consider all data regardless of difficulty or pattern
            d_or_p_list = [None]

        # Loop through each difficulty_or_pattern_of_interest in the list
        for d_or_p in d_or_p_list:
            # If d_or_p is None, skip filtering on difficulty_or_pattern_value
            if d_or_p is None:
                df_filtered_extended = df[df["tag"].isin(tags)]
            else:
                # Filter the dataframe for the tags and difficulty_or_pattern_of_interest
                df_filtered_extended = df[
                    (df["tag"].isin(tags))
                    & (
                        df["difficulty_or_pattern_value"].isin(
                            [d_or_p if d_or_p is not None else str(d_or_p)]
                        )
                    )
                ]

            # Verify the filtered data
            if df_filtered_extended.empty:
                print(
                    f"No data found for dataset: {dataset_name}, tags: {tags} and difficulty/pattern values: {d_or_p}"
                )
                continue  # Skip to the next iteration if there's no data

            # Get the task-specific criteria for this dataset
            dataset_task_criteria = task_tag_criteria.get(dataset_name, {})

            # Process each task according to its important tags and criteria
            for task in df_filtered_extended["task"].unique():
                # Get the important tags for this task in the current dataset (list of tags)
                important_tags = dataset_task_criteria.get(task, [])

                if not important_tags:
                    print(
                        f"No valid important tags specified for task: {task} in dataset: {dataset_name}"
                    )
                    continue

                winning_id = None

                # Loop over each important tag to determine the best `id`
                for tag in important_tags:
                    if tag not in tags:
                        print(
                            f"Tag {tag} not found in tags for dataset: {dataset_name}"
                        )
                        continue

                    # Filter data for the current task and tag
                    df_task_tag = df_filtered_extended[
                        (df_filtered_extended["task"] == task)
                        & (df_filtered_extended["tag"] == tag)
                    ]

                    if df_task_tag.empty:
                        print(
                            f"No data found for task: {task} and tag: {tag} in dataset: {dataset_name}"
                        )
                        continue

                    # Group the data by task, id, difficulty/pattern, and tag, then calculate the mean and std value over all steps
                    df_grouped = df_task_tag.groupby(
                        ["difficulty_or_pattern_value", "task", "id", "tag"],
                        as_index=False,
                    ).agg(mean_value=("value", "mean"), std_value=("value", "std"))

                    # Determine whether to find max or min based on the tag criteria
                    if tag_criteria.get(tag) == "max":
                        idx = df_grouped.groupby(
                            ["difficulty_or_pattern_value", "task"]
                        )["mean_value"].idxmax()
                    elif tag_criteria.get(tag) == "min":
                        idx = df_grouped.groupby(
                            ["difficulty_or_pattern_value", "task"]
                        )["mean_value"].idxmin()
                    else:
                        print(f"Tag {tag} does not have a valid criteria (max/min)")
                        continue

                    # Get the winning row for this task and tag
                    df_winning_row = df_grouped.loc[idx]

                    # If this is the first tag or the same id is confirmed, set the winning id
                    if winning_id is None:
                        winning_id = int(df_winning_row["id"].iloc[0])
                    elif winning_id != int(df_winning_row["id"]):
                        print(
                            f"Inconsistent winning ids for task: {task} in dataset: {dataset_name}"
                        )
                        winning_id = None
                        break

                if winning_id is None:
                    print(
                        f"No consistent winning id found for task: {task} in dataset: {dataset_name}"
                    )
                    continue

                # Get the mean and std values for all tags for this winning id across all steps
                df_winning_data = df_filtered_extended[
                    (df_filtered_extended["task"] == task)
                    & (df_filtered_extended["id"] == winning_id)
                ]

                # Group the data by tag, difficulty, and id to compute the mean and std over all steps
                df_mean_values = df_winning_data.groupby(
                    ["difficulty_or_pattern_value", "task", "id", "tag"], as_index=False
                ).agg(mean_value=("value", "mean"), std_value=("value", "std"))

                # Convert the tuple key to a string representation
                winning_difficulty_or_pattern_value = df_winning_row[
                    "difficulty_or_pattern_value"
                ].iloc[0]
                key = f"{int(task)}_{winning_difficulty_or_pattern_value if winning_difficulty_or_pattern_value != None else None}"

                # Store the mean and std values for all tags
                grouped_dict[key] = {
                    "winning_id": winning_id,
                    "important_tags": important_tags,
                    "mean_values": df_mean_values.to_dict(
                        orient="records"
                    ),  # Mean and std values across all tags
                }

        # Add the current dataset results to the final dictionary
        final_dict[f"{dataset_name}"] = grouped_dict

        # Display the resulting dictionary for this dataset
        # print(
        #     f"Results for dataset: {dataset_name}, tags: {tags} and difficulty/pattern values: {d_or_p_list}"
        # )
        # print(grouped_dict)

    # Dump the final_dict to a JSON file (now with string keys)
    with open(output_json_path, "w") as json_file:
        json.dump(final_dict, json_file, indent=4)

    # print(f"Final results have been written to {output_json_path}")


tags_of_interest = [
    [
        "Environment/Cumulative Reward",
        "WindFarmControl/Individual Performance",
        "WindFarmControl/Avoid Damage Reward",
    ],
    [
        "Environment/Cumulative Reward",
        "WildfireResourceManagement/Reward for Moving Resources to Neighbours",
        "WildfireResourceManagement/Reward for Moving Resources to Self",
        "WildfireResourceManagement/Collective Performance",
        "WildfireResourceManagement/Individual Performance",
    ],
    [
        "Environment/Cumulative Reward",
        "DroneBasedReforestation/Tree Drop Count",
        "DroneBasedReforestation/Recharge Energy Count",
        "DroneBasedReforestation/Save Location Count",
        "DroneBasedReforestation/Out of Energy Count",
        "DroneBasedReforestation/Cumulative Distance Until Tree Drop",
        "DroneBasedReforestation/Cumulative Tree Drop Reward",
        "DroneBasedReforestation/Cumulative Distance Reward",
        "DroneBasedReforestation/Cumulative Normalized Distance Until Tree Drop",
        "DroneBasedReforestation/Cumulative Distance to Existing Trees",
        "DroneBasedReforestation/Highest Potential Soild Found",
        "DroneBasedReforestation/Highest Point on Terrain Found",
        "DroneBasedReforestation/Furthest Distance Explored",
    ],
    [
        "Environment/Cumulative Reward",
        "OceanPlasticCollector/Global Reward",
        "OceanPlasticCollector/Local Reward",
    ],
    [
        "Environment/Cumulative Reward",
        "AerialWildfireSuppression/Water Drop",
        "AerialWildfireSuppression/Water Pickup",
        "AerialWildfireSuppression/Preparing Trees Reward",
        "AerialWildfireSuppression/Preparing Trees",
        "AerialWildfireSuppression/Extinguishing Trees Reward",
        "AerialWildfireSuppression/Extinguishing Trees",
        "AerialWildfireSuppression/Fire Out",
        "AerialWildfireSuppression/Fire too Close to City",
        "AerialWildfireSuppression/Crash Count",
    ],
]

difficulty_or_pattern_of_interest = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    None,
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
]

tag_criteria = {
    "WindFarmControl/Individual Performance": "max",
    "Environment/Cumulative Reward": "max",
    "WindFarmControl/Avoid Damage Reward": "max",
    "WildfireResourceManagement/Reward for Moving Resources to Neighbours": "max",
    "WildfireResourceManagement/Reward for Moving Resources to Self": "max",
    "WildfireResourceManagement/Collective Performance": "max",
    "WildfireResourceManagement/Individual Performance": "max",
    "DroneBasedReforestation/Tree Drop Count": "max",
    "DroneBasedReforestation/Recharge Energy Count": "max",
    "DroneBasedReforestation/Save Location Count": "max",
    "DroneBasedReforestation/Out of Energy Count": "min",
    "DroneBasedReforestation/Cumulative Distance Until Tree Drop": "max",
    "DroneBasedReforestation/Cumulative Tree Drop Reward": "max",
    "DroneBasedReforestation/Cumulative Distance Reward": "max",
    "DroneBasedReforestation/Cumulative Normalized Distance Until Tree Drop": "max",
    "DroneBasedReforestation/Cumulative Distance to Existing Trees": "min",
    "DroneBasedReforestation/Highest Potential Soild Found": "max",
    "DroneBasedReforestation/Highest Point on Terrain Found": "max",
    "DroneBasedReforestation/Furthest Distance Explored": "max",
    "OceanPlasticCollector/Global Reward": "max",
    "OceanPlasticCollector/Local Reward": "max",
    "AerialWildfireSuppression/Water Drop": "max",
    "AerialWildfireSuppression/Water Pickup": "max",
    "AerialWildfireSuppression/Preparing Trees Reward": "max",
    "AerialWildfireSuppression/Preparing Trees": "max",
    "AerialWildfireSuppression/Extinguishing Trees Reward": "max",
    "AerialWildfireSuppression/Extinguishing Trees": "max",
    "AerialWildfireSuppression/Fire Out": "max",
    "AerialWildfireSuppression/Fire too Close to City": "min",
    "AerialWildfireSuppression/Crash Count": "min",
}

task_tag_criteria = {
    "WindFarmControl": {
        0: ["WindFarmControl/Individual Performance"],
        1: ["WindFarmControl/Avoid Damage Reward"],
    },
    "WildfireResourceManagement": {
        0: ["Environment/Cumulative Reward"],
        1: ["WildfireResourceManagement/Reward for Moving Resources to Self"],
        2: ["WildfireResourceManagement/Reward for Moving Resources to Neighbours"],
    },
    "DroneBasedReforestation": {
        0: ["Environment/Cumulative Reward"],
        1: ["Environment/Cumulative Reward"],
        2: ["DroneBasedReforestation/Recharge Energy Count"],
        3: ["Environment/Cumulative Reward"],
        4: ["DroneBasedReforestation/Highest Potential Soild Found"],
        5: ["DroneBasedReforestation/Highest Point on Terrain Found"],
        6: ["DroneBasedReforestation/Furthest Distance Explored"],
    },
    "OceanPlasticCollector": {
        0: ["Environment/Cumulative Reward"],
        1: ["Environment/Cumulative Reward"],
        2: ["Environment/Cumulative Reward"],
        3: ["Environment/Cumulative Reward"],
    },
    "AerialWildfireSuppression": {
        0: ["Environment/Cumulative Reward"],
        1: ["AerialWildfireSuppression/Extinguishing Trees Reward"],
        2: ["AerialWildfireSuppression/Preparing Trees Reward"],
        3: ["AerialWildfireSuppression/Fire Out"],
        4: ["AerialWildfireSuppression/Fire too Close to City"],
        5: ["AerialWildfireSuppression/Water Pickup"],
        6: ["AerialWildfireSuppression/Water Drop"],
        7: ["Environment/Cumulative Reward"],
        8: ["Environment/Cumulative Reward"],
    },
}

path_prefix = "../hivex-results/results/"

test_dirs = [
    f"{path_prefix}WindFarmControl/test",
    f"{path_prefix}WildfireResourceManagement/test",
    f"{path_prefix}DroneBasedReforestation/test",
    f"{path_prefix}OceanPlasticCollection/test",
    f"{path_prefix}AerialWildfireSuppression/test",
]

output_json_path = "best_models_per_task_difficulty_id.json"

datasets = {}

for test_root_dir in test_dirs:
    test_data = list_directories(test_root_dir, "test")
    key = test_root_dir.split("/")[-2]
    datasets[key] = test_data

# Run the function
find_best_models(
    datasets,
    tags_of_interest,
    difficulty_or_pattern_of_interest,
    output_json_path,
    tag_criteria,
    task_tag_criteria,
)
