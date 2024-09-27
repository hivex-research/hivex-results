# HIVEX Baseline Results

This repository holds all result plots and the script to create such. Due to size constraints, unfortunately the checkpoints and results from all train and test runs can only be viewed and downloaded on google-drive: [Hivex Baseline Results](https://drive.google.com/drive/folders/1vOvnMtlQL0zSWivlUKA1oZAh-mineogP?usp=drive_link). This includes baseline results for:

- Wind Farm Control: [Results](https://drive.google.com/drive/folders/1PtSoEQP938Flp71MLC6nUyeVN2ERckWq?usp=drive_link)
- Wildfire Resource Management: [Results](https://drive.google.com/drive/folders/1WHpKUqgS2fiOE_th8D5A3-0HKl2jun7D?usp=drive_link)
- Drone-Based Reforestation: [Results](https://drive.google.com/drive/folders/1EjgTa2SCqpCBcmgLkscflcvX8X8U4FV7?usp=drive_link)
- Ocean Plastic Collection: [Results](https://drive.google.com/drive/folders/1Okbds515W5JEo3ZXdMmJjJJyLsoUo_ZE?usp=drive_link)
- Aerial Wildfire Suppression: [Results](https://drive.google.com/drive/folders/1ZmBOROxylijbZxLcKppMy4y75BK45e2a?usp=drive_link)

## Overview of Featured Average Metrics

<br>
<div align="center">
  <img src="docs\images\cumulative_reward_multiple.svg"
      style="border-radius:10px"
      alt="hivex baseline results average"/>
</div>
<br>
<br>

# ✨ Submit your own Results to the <!--- [HIVEX Leaderboard](https://huggingface.co/spaces/hivex-research/hivex-leaderboard) --> on Huggingface 🤗

1. Install all dependencies as described in the <!---  [HIVEX repository README](https://github.com/hivex-research/hivex/tree/main). -->

2. Run the Train and Test Pipeline in the <!---  [HIVEX repository](https://github.com/hivex-research/hivex/tree/main) -->, either using <!--- [ML-Agents](https://github.com/hivex-research/hivex/tree/main?tab=readme-ov-file#-reproducing-paper-results) --> or with your <!--- [favorite framework](https://github.com/hivex-research/hivex/tree/main?tab=readme-ov-file#-additional-environments-and-training-frameworks). -->

3. Add your results to the respective environment/train and environment/test folders. We have provided a `train_dummy_folder` (results/WindFarmControl/train_dummy_folder) and `test_dummy_folder` (results/WindFarmControl/test_dummy_folder) with results for training and testing on the Wind Farm Control environment.

4. Run `find_best_models.py`

This script generates data from your results. We provide the data from the paper runs for your to look at as an example: `tools/huggingface/best_models_per_task_difficulty_id.json`

```shell
python tools/huggingface/find_best_models.py
```

5. Run `generate_hf_yaml.py`

Uncomment the environment data parser you need for your data. For example for our dummy data we need `generate_yaml_WFC(data["WindFarmControl"], key)`. This script takes the data generated in the previous step and turns it into folders including the checkpoint etc. of your training run and a `README.md`, which serves as the model card including important meta-data that is needed for the automatic fetching of the leaderboard of your model.

```shell
python tools/huggingface/generate_hf_yaml.py
```

6. Finally, upload the content of the generated folder(s) to Huggingface 🤗 as a new model.

7. Every 24 hours, the <!--- [HIVEX Leaderboard](https://huggingface.co/spaces/hivex-research/hivex-leaderboard) --> is fetching new models. We will review your model as soon as possible and it to the verified list of models as soon as possible. If you have any questions please feel free to reach out to <!--- p.d.siedler@gmail.com. -->
