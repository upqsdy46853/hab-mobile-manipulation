# Mobile Manipulation for the Home Assistant Benchmark (HAB)

This is a PyTorch implementation of:

[Multi-skill Mobile Manipulation for Object Rearrangement](https://arxiv.org/abs/2209.02778)<br/>
Jiayuan Gu, Devendra Singh Chaplot, Hao Su, Jitendra Malik<br/>
UC San Diego, Meta AI Research, UC Berkeley

Project website: <https://sites.google.com/view/hab-m3>


https://user-images.githubusercontent.com/17827258/189198353-9733887a-f7ad-4efc-a927-0b6d86f8d7b0.mp4



**Table of Contents**

- [Installation](#installation)
- [Data](#data)
- [Interactive play](#interactive-play)
- [Evaluation](#evaluation)
  - [Evaluate a sub-task](#evaluate-a-sub-task)
  - [Evaluate a HAB (Home Assistant Benchmark) task](#evaluate-a-hab-home-assistant-benchmark-task)
- [Training](#training)
- [Acknowledgments](#acknowledgments)

## Installation

```bash
# Ensure the latest submodules
git submodule update --init --recursive
# Create a conda env
conda create -n hab-mm python=3.7
# Activate the conda env
conda activate hab-mm
# Install habitat-sim from source
conda install cmake=3.14.0 patchelf ninja
cd habitat-sim && pip install -r requirements.txt && python setup.py install --bullet --headless && cd ..
# Install habitat-lab
cd habitat-lab && pip install -r requirements.txt && python setup.py develop && cd ..
# Install requirements
pip install -r requirements.txt
# Install habitat manipulation
python setup.py develop
# Post-installation
echo "export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet" >> ~/.bashrc
```

We also provide a docker image: `docker pull jiayuangu/hab-mm`.

---

**Known Issues**

If you encounter any memory leak during training, please try to install habitat-sim from conda.

```bash
# Install pytorch first
conda install -y pytorch==1.5.1 cudatoolkit=10.2 -c pytorch
# Install habitat-sim from aihabitat-nightly
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat-nightly
# Install habitat-sim from source again
cd habitat-sim && pip install -r requirements.txt && python setup.py install --bullet --headless && cd ..
```

<details>
  <summary>Troubleshooting</summary>
  
- [Could not find an EGL device for CUDA device 0](https://github.com/facebookresearch/habitat-sim/issues/288): reinstall Nvidia driver

</details>

## Data

```bash
# Download ReplicaCAD v1.4, YCB objects, and Fetch URDF.
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
# Generate physical config to correctly configure the simulator backend
python -c "from habitat.datasets.utils import check_and_gen_physics_config; check_and_gen_physics_config()"
# Download generated episodes
pip install gdown
gdown https://drive.google.com/drive/folders/1oEhsiqoWcEA2FNuQd9QfCPKNKgSwHbaW -O data/datasets/rearrange/v3 --folder
```

To re-generate our episodes, please refer to [episode generation](INSTRUCTIONS.md#episode-generation).

## Interactive play

Interactively play the task with the default config:

```bash
python habitat_extensions/tasks/rearrange/play.py
```

Use `i/j/k/l` to move the robot end-effector, and `w/a/s/d` to move the robot base. Use `f/g` to grasp or release an object.

---

Use a specific task config:

```bash
python habitat_extensions/tasks/rearrange/play.py  --cfg configs/rearrange/tasks/pick_v1.yaml
```

Or use a specific RL config:

```bash
python habitat_extensions/tasks/rearrange/play.py  --cfg configs/rearrange/skills/tidy_house/pick_v1_joint_SCR.yaml
```

## Evaluation

```bash
# Evaluate the latest checkpoint of a skill saved at "data/results/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR"
python mobile_manipulation/run_ppo.py --cfg configs/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR.yaml --run-type eval
# Evaluate the latest checkpoint of a skill saved at "data/results/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR/seed=100"
python mobile_manipulation/run_ppo.py --cfg configs/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR.yaml --run-type eval PREFIX seed=101 TASK_CONFIG.SEED 101
```

## Training

```bash
# The result will be saved at "data/results/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR".
python mobile_manipulation/run_ppo.py --cfg configs/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR.yaml --run-type train
# Specify a prefix and a random seed for the experiment.
# The result will be saved at "data/results/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR/seed=101" 
python mobile_manipulation/run_ppo.py --cfg configs/rearrange/skills/tidy_house/mono_v1_ppo_v0_SCR.yaml --run-type train PREFIX seed=101 TASK_CONFIG.SEED 101
```

## Acknowledgments

This repository is inspired by [Habitat Lab](https://github.com/facebookresearch/habitat-lab) for RL environments and PPO implementation. We would also like to thank [Andrew Szot](https://www.andrewszot.com/) and [Alexander Clegg](https://scholar.google.com/citations?user=p463opcAAAAJ&hl=en) for their help in using Habitat 2.0.
