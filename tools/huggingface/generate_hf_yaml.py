import yaml
import json
import os
import shutil


def generate_yaml_WFC(data, key):
    # Extract pattern and task from the key
    task, pattern = map(int, key.split("_"))
    wind_farm = data[key]
    winning_id = wind_farm["winning_id"]
    mean_values = wind_farm["mean_values"]
    important_tags = wind_farm["important_tags"][0]

    # Prepare the basic structure
    result = {
        "library_name": "hivex",
        "original_train_name": f"WindFarmControl_pattern_{pattern}_task_{task}_run_id_{winning_id}_train",
        "tags": [
            "hivex",
            "hivex-wind-farm-control",
            "reinforcement-learning",
            "multi-agent-reinforcement-learning",
        ],
        "model-index": [
            {
                "name": f"hivex-WFC-PPO-baseline-task-{task}-pattern-{pattern}",
                "results": [
                    {
                        "task": {
                            "type": ("main-task" if task == 0 else "sub-task"),
                            "name": ("main_task" if task == 0 else "avoid_damage"),
                            "task-id": task,
                            "pattern-id": pattern,
                        },
                        "dataset": {
                            "name": "hivex-wind-farm-control",
                            "type": "hivex-wind-farm-control",
                        },
                        "metrics": [
                            (
                                {
                                    "type": "cumulative_reward",
                                    "value": f"{v['mean_value']} +/- {v['std_value']}",
                                    "name": "Cumulative Reward",
                                    "verified": True,
                                }
                                if "Cumulative Reward" in v["tag"]
                                else (
                                    {
                                        "type": "individual_performance",
                                        "value": f"{v['mean_value']} +/- {v['std_value']}",
                                        "name": "Individual Performance",
                                        "verified": True,
                                    }
                                    if "Individual Performance" in v["tag"]
                                    else {
                                        "type": "avoid_damage_reward",
                                        "value": f"{v['mean_value']} +/- {v['std_value']}",
                                        "name": "Avoid Damage Reward",
                                        "verified": True,
                                    }
                                )
                            )
                            for v in mean_values
                        ],
                    }
                ],
            }
        ],
    }

    # Convert the result to YAML format
    results = yaml.dump(result, sort_keys=False)

    # Define the directory and file paths
    directory = f"hf_yaml_files/hivex-WFC-PPO-baseline-task-{task}-pattern-{pattern}"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "README.md")

    # Save to README.md in the corresponding directory
    with open(file_path, "w") as file:
        file.write("---\n")
        file.write(results)
        file.write("---")

        # Add the descriptive section
        file.write(
            f"This model serves as the baseline for the **Wind Farm Control** environment, "
            f"trained and tested on task <code>{task}</code> with pattern <code>{pattern}</code> using the Proximal Policy "
            f"Optimization (PPO) algorithm.<br><br>"
            "Environment: **Wind Farm Control**<br>"
            f"Task: <code>{task}</code><br>"
            f"Pattern: <code>{pattern}</code><br>"
            "Algorithm: <code>PPO</code><br>"
            "Episode Length: <code>5000</code><br>"
            "Training <code>max_steps</code>: <code>8000000</code><br>"
            "Testing <code>max_steps</code>: <code>8000000</code><br><br>"
            "Train & Test [Scripts](https://github.com/hivex-research/hivex)<br>"
            "Download the [Environment](https://github.com/hivex-research/hivex-environments)"
        )

    # Copy the contents of the folder corresponding to the original_train_name
    original_train_name = result["original_train_name"]
    source_dir = os.path.join(
        "results", "WindFarmControl", "train", original_train_name
    )
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(directory, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)
    else:
        print(f"Warning: Source directory '{source_dir}' does not exist.")

    return results


def generate_yaml_OPC(data, key):
    # Extract task and difficulty from the key
    task, difficulty = map(int, key.split("_"))
    wildfire = data[key]
    winning_id = wildfire["winning_id"]
    mean_values = wildfire["mean_values"]

    # Map task ID to task name
    task_name_map = {
        0: "main_task",
        1: "find_highest_polluted_area",
        2: "group_up",
        3: "avoid_plastic",
    }

    # Prepare the basic structure
    result = {
        "library_name": "hivex",
        "original_train_name": f"OceanPlasticCollection_task_{task}_run_id_{winning_id}_train",
        "tags": [
            "hivex",
            "hivex-ocean-plastic-collection",
            "reinforcement-learning",
            "multi-agent-reinforcement-learning",
        ],
        "model-index": [
            {
                "name": f"hivex-OPC-PPO-baseline-task-{task}",
                "results": [
                    {
                        "task": {
                            "type": ("main-task" if task == 0 else "sub-task"),
                            "name": task_name_map.get(task),
                            "task-id": task,
                        },
                        "dataset": {
                            "name": "hivex-ocean-plastic-collection",
                            "type": "hivex-ocean-plastic-collection",
                        },
                        "metrics": [
                            {
                                "type": v["tag"]
                                .split("/")[1]
                                .lower()
                                .replace(" ", "_")
                                .replace("/", "_"),
                                "value": f"{v['mean_value']} +/- {v['std_value']}",
                                "name": v["tag"].split("/")[-1],
                                "verified": True,
                            }
                            for v in mean_values
                        ],
                    }
                ],
            }
        ],
    }

    # Convert the result to YAML format
    results = yaml.dump(result, sort_keys=False)

    # Define the directory and file paths
    directory = f"hf_yaml_files/hivex-OPC-PPO-baseline-task-{task}"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "README.md")

    # Save to README.md in the corresponding directory
    with open(file_path, "w") as file:
        file.write("---\n")
        file.write(results)
        file.write("---")

        # Add the descriptive section
        file.write(
            f"This model serves as the baseline for the **Ocean Plastic Collection** environment, "
            f"trained and tested on task <code>{task}</code> using the Proximal Policy "
            f"Optimization (PPO) algorithm.<br><br>"
            "Environment: **Ocean Plastic Collection**<br>"
            f"Task: <code>{task}</code><br>"
            "Algorithm: <code>PPO</code><br>"
            "Episode Length: <code>5000</code><br>"
            "Training <code>max_steps</code>: <code>3000000</code><br>"
            "Testing <code>max_steps</code>: <code>150000</code><br><br>"
            "Train & Test [Scripts](https://github.com/hivex-research/hivex)<br>"
            "Download the [Environment](https://github.com/hivex-research/hivex-environments)"
        )

    # Copy the contents of the folder corresponding to the original_train_name
    original_train_name = result["original_train_name"]
    source_dir = os.path.join(
        "results", "OceanPlasticCollection", "train", original_train_name
    )
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(directory, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)
    else:
        print(f"Warning: Source directory '{source_dir}' does not exist.")

    return results


def generate_yaml_WRM(data, key):
    # Extract task and difficulty from the key
    task, difficulty = map(int, key.split("_"))
    wildfire = data[key]
    winning_id = wildfire["winning_id"]
    mean_values = wildfire["mean_values"]
    important_tags = wildfire["important_tags"][0]

    # Map task ID to task name
    task_name_map = {0: "main_task", 1: "keep_all", 2: "distribute_all"}

    # Prepare the basic structure
    result = {
        "library_name": "hivex",
        "original_train_name": f"WildfireResourceManagement_difficulty_{difficulty}_task_{task}_run_id_{winning_id}_train",
        "tags": [
            "hivex",
            "hivex-wildfire-resource-management",
            "reinforcement-learning",
            "multi-agent-reinforcement-learning",
        ],
        "model-index": [
            {
                "name": f"hivex-WRM-PPO-baseline-task-{task}-difficulty-{difficulty}",
                "results": [
                    {
                        "task": {
                            "type": ("main-task" if task == 0 else "sub-task"),
                            "name": task_name_map.get(task),
                            "task-id": task,
                            "difficulty-id": difficulty,
                        },
                        "dataset": {
                            "name": "hivex-wildfire-resource-management",
                            "type": "hivex-wildfire-resource-management",
                        },
                        "metrics": [
                            {
                                "type": v["tag"]
                                .split("/")[1]
                                .lower()
                                .replace(" ", "_")
                                .replace("/", "_"),
                                "value": f"{v['mean_value']} +/- {v['std_value']}",
                                "name": v["tag"].split("/")[-1],
                                "verified": True,
                            }
                            for v in mean_values
                        ],
                    }
                ],
            }
        ],
    }

    # Convert the result to YAML format
    results = yaml.dump(result, sort_keys=False)

    # Define the directory and file paths
    directory = (
        f"hf_yaml_files/hivex-WRM-PPO-baseline-task-{task}-difficulty-{difficulty}"
    )
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "README.md")

    # Save to README.md in the corresponding directory
    with open(file_path, "w") as file:
        file.write("---\n")
        file.write(results)
        file.write("---")

        # Add the descriptive section
        file.write(
            f"This model serves as the baseline for the **Wildfire Resource Management** environment, "
            f"trained and tested on task <code>{task}</code> with difficulty <code>{difficulty}</code> using the Proximal Policy "
            f"Optimization (PPO) algorithm.<br><br>"
            "Environment: **Wildfire Resource Management**<br>"
            f"Task: <code>{task}</code><br>"
            f"Difficulty: <code>{difficulty}</code><br>"
            "Algorithm: <code>PPO</code><br>"
            "Episode Length: <code>500</code><br>"
            "Training <code>max_steps</code>: <code>450000</code><br>"
            "Testing <code>max_steps</code>: <code>45000</code><br><br>"
            "Train & Test [Scripts](https://github.com/hivex-research/hivex)<br>"
            "Download the [Environment](https://github.com/hivex-research/hivex-environments)"
        )

    # Copy the contents of the folder corresponding to the original_train_name
    original_train_name = result["original_train_name"]
    source_dir = os.path.join(
        "results", "WildfireResourceManagement", "train", original_train_name
    )
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(directory, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)
    else:
        print(f"Warning: Source directory '{source_dir}' does not exist.")

    return results


def generate_yaml_DBR(data, key):
    # Extract task and difficulty from the key
    task, difficulty = map(int, key.split("_"))
    drone = data[key]
    winning_id = drone["winning_id"]
    mean_values = drone["mean_values"]
    important_tags = drone["important_tags"][0]

    # Map task ID to task name
    task_name_map = {
        0: "main_task",
        1: "find_closest_forest_perimeter",
        2: "pick_up_seed_at_base",
        3: "drop_seed",
        4: "find_highest_potential_seed_drop_location",
        5: "find_highest_potential_seed_drop_location",  # Repeated for task 5 based on your example
        6: "explore_furthest_distance_and_return_to_base",
    }

    # Prepare the basic structure
    result = {
        "library_name": "hivex",
        "original_train_name": f"DroneBasedReforestation_difficulty_{difficulty}_task_{task}_run_id_{winning_id}_train",
        "tags": [
            "hivex",
            "hivex-drone-based-reforestation",
            "reinforcement-learning",
            "multi-agent-reinforcement-learning",
        ],
        "model-index": [
            {
                "name": f"hivex-DBR-PPO-baseline-task-{task}-difficulty-{difficulty}",
                "results": [
                    {
                        "task": {
                            "type": ("main-task" if task == 0 else "sub-task"),
                            "name": task_name_map.get(task),
                            "task-id": task,
                            "difficulty-id": difficulty,
                        },
                        "dataset": {
                            "name": "hivex-drone-based-reforestation",
                            "type": "hivex-drone-based-reforestation",
                        },
                        "metrics": [
                            {
                                "type": v["tag"]
                                .split("/")[-1]
                                .lower()
                                .replace(" ", "_")
                                .replace("/", "_"),
                                "value": f"{v['mean_value']} +/- {v['std_value']}",
                                "name": v["tag"].split("/")[-1],
                                "verified": True,
                            }
                            for v in mean_values
                            if v["mean_value"] != 0.0 or v["std_value"] != 0.0
                        ],
                    }
                ],
            }
        ],
    }

    # Convert the result to YAML format
    results = yaml.dump(result, sort_keys=False)

    # Define the directory and file paths
    directory = (
        f"hf_yaml_files/hivex-DBR-PPO-baseline-task-{task}-difficulty-{difficulty}"
    )
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "README.md")

    # Save to README.md in the corresponding directory
    with open(file_path, "w") as file:
        file.write("---\n")
        file.write(results)
        file.write("---\n\n")

        # Add the descriptive section
        file.write(
            f"This model serves as the baseline for the **Drone-Based Reforestation** environment, "
            f"trained and tested on task <code>{task}</code> with difficulty <code>{difficulty}</code> using the Proximal Policy "
            f"Optimization (PPO) algorithm.<br><br>"
            "Environment: **Drone-Based Reforestation**<br>"
            f"Task: <code>{task}</code><br>"
            f"Difficulty: <code>{difficulty}</code><br>"
            "Algorithm: <code>PPO</code><br>"
            "Episode Length: <code>2000</code><br>"
            "Training <code>max_steps</code>: <code>1200000</code><br>"
            "Testing <code>max_steps</code>: <code>300000</code><br><br>"
            "Train & Test [Scripts](https://github.com/hivex-research/hivex)<br>"
            "Download the [Environment](https://github.com/hivex-research/hivex-environments)"
        )

    # Copy the contents of the folder corresponding to the original_train_name
    original_train_name = result["original_train_name"]
    source_dir = os.path.join(
        "results", "DroneBasedReforestation", "train", original_train_name
    )
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(directory, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)
    else:
        print(f"Warning: Source directory '{source_dir}' does not exist.")

    return results


def generate_yaml_AWS(data, key):
    # Extract task and difficulty from the key
    task, difficulty = map(int, key.split("_"))
    aerial = data[key]
    winning_id = aerial["winning_id"]
    mean_values = aerial["mean_values"]
    important_tags = aerial["important_tags"][0]

    # Map task ID to task name for AerialWildfireSuppression
    task_name_map = {
        0: "main_task",
        1: "maximize_extinguished_burning_trees",
        2: "maximize_preparing_non_burning_trees",
        3: "minimize_time_fire_burning",
        4: "protect_village",
        5: "pick_up_water",
        6: "drop_water",
        7: "find_fire",
        8: "find_village",
    }

    # Prepare the basic structure
    result = {
        "library_name": "hivex",
        "original_train_name": f"AerialWildfireSuppression_difficulty_{difficulty}_task_{task}_run_id_{winning_id}_train",
        "tags": [
            "hivex",
            "hivex-aerial-wildfire-suppression",
            "reinforcement-learning",
            "multi-agent-reinforcement-learning",
        ],
        "model-index": [
            {
                "name": f"hivex-AWS-PPO-baseline-task-{task}-difficulty-{difficulty}",
                "results": [
                    {
                        "task": {
                            "type": ("main-task" if task == 0 else "sub-task"),
                            "name": task_name_map.get(task, "unknown_task"),
                            "task-id": task,
                            "difficulty-id": difficulty,
                        },
                        "dataset": {
                            "name": "hivex-aerial-wildfire-suppression",
                            "type": "hivex-aerial-wildfire-suppression",
                        },
                        "metrics": [
                            {
                                "type": v["tag"]
                                .split("/")[-1]
                                .lower()
                                .replace(" ", "_")
                                .replace("/", "_"),
                                "value": f"{v['mean_value']} +/- {v['std_value']}",
                                "name": v["tag"].split("/")[-1],
                                "verified": True,
                            }
                            for v in mean_values
                            if v["mean_value"] != 0.0 or v["std_value"] != 0.0
                        ],
                    }
                ],
            }
        ],
    }

    # Convert the result to YAML format
    results = yaml.dump(result, sort_keys=False)

    # Define the directory and file paths
    directory = (
        f"hf_yaml_files/hivex-AWS-PPO-baseline-task-{task}-difficulty-{difficulty}"
    )
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "README.md")

    # Save to README.md in the corresponding directory
    with open(file_path, "w") as file:
        file.write("---\n")
        file.write(results)
        file.write("---\n\n")

        # Add the descriptive section
        file.write(
            f"This model serves as the baseline for the **Aerial Wildfire Suppression** environment, "
            f"trained and tested on task <code>{task}</code> with difficulty <code>{difficulty}</code> using the Proximal Policy "
            f"Optimization (PPO) algorithm.<br><br>\n\n"
            "Environment: **Aerial Wildfire Suppression**<br>\n"
            f"Task: <code>{task}</code><br>\n"
            f"Difficulty: <code>{difficulty}</code><br>\n"
            "Algorithm: <code>PPO</code><br>\n"
            "Episode Length: <code>3000</code><br>\n"
            "Training <code>max_steps</code>: <code>1800000</code><br>\n"
            "Testing <code>max_steps</code>: <code>180000</code><br><br>\n\n"
            "Train & Test [Scripts](https://github.com/hivex-research/hivex)<br>\n"
            "Download the [Environment](https://github.com/hivex-research/hivex-environments)"
        )

    # Copy the contents of the folder corresponding to the original_train_name
    original_train_name = result["original_train_name"]
    source_dir = os.path.join(
        "results", "AerialWildfireSuppression", "train", original_train_name
    )
    if os.path.exists(source_dir):
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(directory, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, target_item)
            else:
                shutil.copy2(source_item, target_item)
    else:
        print(f"Warning: Source directory '{source_dir}' does not exist.")

    return results


# first run tools/huggingface/find_best_models.py to generate this file
data = json.load(open("best_models_per_task_difficulty_id.json"))

# for key in data["WindFarmControl"]:
#     print(generate_yaml_WFC(data["WindFarmControl"], key))

# for key in data["WildfireResourceManagement"]:
#     print(generate_yaml_WRM(data["WildfireResourceManagement"], key))

# for key in data["DroneBasedReforestation"]:
#     print(generate_yaml_DBR(data["DroneBasedReforestation"], key))

# for key in data["OceanPlasticCollection"]:
#     print(generate_yaml_OPC(data["OceanPlasticCollection"], key))

for key in data["AerialWildfireSuppression"]:
    print(generate_yaml_AWS(data["AerialWildfireSuppression"], key))
