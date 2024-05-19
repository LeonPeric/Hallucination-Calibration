# Hallucination-Calibration

## Setup Instructions

Follow these steps to set up the project environment and install the required dependencies.

### Step 1: Install Editable Package

First, navigate to the `src/minGPT/` directory and install the package in editable mode.

```sh
cd src/minGPT/
pip install -e .
```
### Step 2: Install Project Dependencies

Return to the root directory of the project and install the required dependencies listed in requirements.txt.

```sh
cd ../..
pip install -r requirements.txt
```

### Step 3: Install PyTorch with CUDA

Install PyTorch along with the desired CUDA version. Here is the command for CUDA 12.1:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Demos
Use fact_dataset_test notebook in ./demos to generate synthetic dataset, train a model and calculate hallucination rates. 

## Experimental Results
Experimental results can be found in ./src/experiment/experiments.json