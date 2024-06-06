
# Stone Classification Project

## Overview

This repository contains code for training, evaluating, and deploying a model to detect stones into conveer. It includes a Python for model development and a Python script for command-line interface (CLI) usage.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Ulatel/https---gitlab.com-LarionovaUliana-stones.git
    cd https---gitlab.com-LarionovaUliana-stones
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Checkout the folder into src directory.
    ```bash
    cd src
    ```

#### Training the Model

1. Load your dataset into src/datasets folder.

2. To train the model using the CLI, run:
    ```bash
    python model.py train --data_path <path_to_data>
    ```
3. Your model will be trained and saved into src/runs folder

#### Evaluating the Model

1. To evaluate the model using the CLI, run:
    ```bash
    python model.py evaluate --model_path <path_to_model>
    ```

#### Predict by Model

1. Load your photo

2. To predict using the CLI, run:
    ```bash
    python model.py predict --path <path_to_image>
    ```

#### Looking the Demo

1. Load your video

2. To predict using the CLI, run:
    ```bash
    python model.py predict_video --path <path_to_video>
    ```

### Resources Used

- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter Notebook](https://jupyter.org/)
.
