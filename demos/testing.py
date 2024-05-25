import sys
import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the path to the src/lib directory to the system path
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/lib'))
if lib_path not in sys.path:
    sys.path.append(lib_path)

from main import DatasetGenerator

mingpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/minGPT'))
if mingpt_path not in sys.path:
    sys.path.append(mingpt_path)

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

def prepare_dataset():
    dataset = DatasetGenerator(
        small=True, samples=50000, number_person=100, max_foods_per_person=30,
        distribution="zipf", place=False, day=False, dataset_folder='dataset/data/', food_global=True
    )
    dataset.generate_probabilities()
    dataset.generate()
    print("Monofact rate: ", dataset.monofact_rate())
    dataset.tokenize()
    train_dataset = [torch.tensor(x, dtype=torch.long) for x in dataset.dataset_tokenized]
    return dataset, train_dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:], dtype=torch.long)
        return x, y

def prepare_model(dataset):
    set_seed(42)
    model_config = GPT.get_default_config()
    model_config.n_layer = 12
    model_config.n_head = 8
    model_config.n_embd = 512
    model_config.vocab_size = dataset.vocabulary_size
    model_config.model_type = None
    model_config.block_size = 2
    model = GPT(model_config)
    return model

def prepare_trainer(model, train_data):
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-5
    train_config.max_iters = 100
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_data)
    return trainer

def train_model(trainer):
    def batch_end_callback(trainer):
        if trainer.iter_num % 1000 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

def generate_sequences(names, model, dataset, n_sequences=1000):
    model.eval()
    collected_generations = []
    for name in names:
        name_tokenized = dataset.word2id[name]
        for _ in tqdm(range(n_sequences)):
            x = torch.Tensor([0, name_tokenized]).unsqueeze(0).long().to("cuda")
            y_gen = model.generate(x, 1, do_sample=True)
            food_item = dataset.decode([y_gen[0][2]])[0]
            collected_generations.append({'name': name, 'food': food_item})
    return collected_generations

def analyze_generations(name, generations_df, dataset):
    melessa_df = generations_df[generations_df['name'] == name]
    food_counts = melessa_df['food'].value_counts()
    food_counts = dict(food_counts / food_counts.sum())
    
    true_food_names = [d.strip() for d in dataset.distributions_per_name[name]['food']]
    true_food_probabilities = dataset.distributions_per_name[name]['prob']
    true_probabilities = dict(zip(true_food_names, true_food_probabilities))

    generated_probabilities = pd.DataFrame(dataset.dataset_splitted)[1].value_counts()
    generated_probabilities = dict(generated_probabilities / generated_probabilities.sum())

    for food in food_counts.keys():
        if food not in true_probabilities:
            true_probabilities[food] = 0
        if food not in generated_probabilities:
            generated_probabilities[food] = 0

    for food in true_probabilities.keys():
        if food not in food_counts:
            food_counts[food] = 0
        if food not in generated_probabilities:
            generated_probabilities[food] = 0

    comparison_df = pd.DataFrame({'food': true_probabilities.keys()})
    comparison_df['generated_distribution'] = comparison_df['food'].apply(lambda x: food_counts[x])
    comparison_df['training_data_distribution'] = comparison_df['food'].apply(lambda x: generated_probabilities[x])
    comparison_df['true_distribution'] = comparison_df['food'].apply(lambda x: true_probabilities[x])
    comparison_df = comparison_df.set_index('food')
    return comparison_df

def plot_distribution_comparison(comparison_df):
    comparison_df.plot.bar(figsize=(16, 4))
    plt.yscale("log")
    plt.axvline(30 - 0.5, color='red')
    plt.show()

def calculate_monofact_rate(dataset):
    generated_foods = pd.Series([str(d) for d in dataset.dataset_splitted])
    monofact_number = np.sum(generated_foods.value_counts() == 1)
    monofact_rate = monofact_number / len(dataset.dataset_splitted)
    return monofact_rate

def main():
    dataset, train_dataset = prepare_dataset()
    train_data = MyDataset(train_dataset)
    model = prepare_model(dataset)
    trainer = prepare_trainer(model, train_data)
    train_model(trainer)
    
    collected_generations = generate_sequences(names, model, dataset)
    generations_df = pd.DataFrame(collected_generations)
    
    comparison_df = analyze_generations(name, generations_df, dataset)
    plot_distribution_comparison(comparison_df)
    
    hallucination_rate = len(generations_df[~generations_df['food'].isin(dataset.distributions_per_name['Melessa']['food'])]) / len(generations_df)
    print(f"Hallucination rate: {hallucination_rate}")
    
    monofact_rate = calculate_monofact_rate(dataset)
    print(f"Monofact rate: {monofact_rate}")

if __name__ == "__main__":
    main()
