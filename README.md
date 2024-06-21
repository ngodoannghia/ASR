# Speech to Text Project

This project aims to train a `wav2vec` model for speech-to-text tasks using both Vietnamese and English data. Here is the structure and setup of the project.

## Project Structure

ASR/
├── configs/
│ ├── config.yaml
│ ├── data/
│ │ └── default.yaml
│ ├── training/
│ │ └── default.yaml
│ ├── wandb/
│ └── default.yaml
├── data/
│ ├── raw/
│ ├── processed/
│ ├── exploration/
├── notebooks/
│ ├── data_exploration.ipynb
│ ├── training_analysis.ipynb
├── scripts/
│ ├── data_preprocessing.py
│ ├── train.py
│ ├── utils.py
├── models/
│ ├── checkpoints/
│ ├── wav2vec_model.py
├── logs/
│ ├── wandb/
├── environment.yml
└── README.md

## Setup

1. **Create and activate the conda environment**:
    ```bash
    conda env create -f environment.yml
    conda activate speech_to_text
    ```

2. **Preprocess data**:
    ```bash
    python scripts/data_preprocessing.py
    ```

3. **Train the model**:
    ```bash
    python scripts/train.py
    ```

4. **Explore data and training results**:
    - Use `notebooks/data_exploration.ipynb` for data exploration.
    - Use `notebooks/training_analysis.ipynb` for analyzing training results.

## Configurations

Configurations are managed using Hydra. Modify the `configs/config.yaml` or create new configurations in the `configs/` directory.

## Logging

We use `wandb` for logging. Ensure you have set up your `wandb` account and updated the `configs/wandb/default.yaml` with your project name and entity.