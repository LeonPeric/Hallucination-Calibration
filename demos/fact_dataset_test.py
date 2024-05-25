import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.lib.fact_dataset_generator import FactDatasetGenerator
from src.minGPT.mingpt.model import GPT
from src.minGPT.mingpt.utils import set_seed
from src.minGPT.mingpt.trainer import Trainer

def load_or_generate_dataset(dataset):
    load_dataset = True
    if load_dataset:
        dataset.load_dataset()
        true_dist = dataset.true_dist 
        training_data = dataset.training_data
    else:
        temp = dataset.generate_all_possibilities()
        true_dist = dataset.generate_true_dist(alpha=alpha)
        training_data = dataset.sample_training_data(training_dataset_size, true_dist.tolist())
    return true_dist, training_data

def create_dataset(train_dataset):
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
            y = torch.tensor(self.data[idx][1:], dtype=torch.long)
            return x, y
    return MyDataset(train_dataset)

def create_model(device, dataset):
    model_config = GPT.get_default_config()
    model_config.n_layer = 12
    model_config.n_head = 8
    model_config.n_embd = 512
    model_config.vocab_size = dataset.vocab_size
    model_config.model_type = None
    model_config.block_size = 3
    model = GPT(model_config).to(device)
    return model

def create_trainer(train_config, model, train_data):
    trainer = Trainer(train_config, model, train_data)
    return trainer

def train_model(seed, model, train_data, trainer, epochs=1):
  with open("training_logs.txt", "w") as log_file:
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        trainer.run()
        current_loss = trainer.loss.item()
        print(f"Training completed for epoch {epoch + 1}")
        log_message = f"Seed: {seed}, Epoch {epoch + 1}, Loss: {current_loss:.5f}\n"
        log_file.write(log_message)

    torch.save(model.state_dict(), dataset.experiment_path[:-5] + f"model_epoch_{epoch + 1}.pt")


def run_training_for_seed(seed):
    # Set the seed for reproducibility
    set_seed(seed)

    # Load dataset
    true_dist_size = 1000
    training_dataset_size = int(0.8 * true_dist_size)
    alpha = 1
    dataset = FactDatasetGenerator(number_person=100, distribution="zipf", dataset_folder='../src/data/',
                                   food_list_name="food_list_small.txt", true_dist_size=true_dist_size,
                                   experiment_path="src/experiment/small_dataset/data/")
    true_dist, training_data = load_or_generate_dataset(dataset)

    # Create datasets
    train_dataset = [torch.tensor(x, dtype=torch.long) for x in dataset.tokenized_training_data]
    train_data = create_dataset(train_dataset)

    # Model configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device, dataset)

    # Trainer configuration
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-5
    train_config.max_iters = 2000
    train_config.num_workers = 0

    # Create trainer object
    trainer = create_trainer(train_config, model, train_data)

    # Train the model
    train_model(seed, model, train_data, trainer, epochs=2)  # set the number of epochs
    model.eval()


# Define a list of seed values to iterate over
seed_list = [42, 123, 456, 789]

# Run training for each seed value
for seed in seed_list:
    print(f"Training for seed: {seed}")
    run_training_for_seed(seed)