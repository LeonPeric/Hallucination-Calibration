import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import json
from src.lib.fact_dataset_generator import FactDatasetGenerator
from src.minGPT.mingpt.model import GPT
from src.minGPT.mingpt.utils import set_seed
from src.minGPT.mingpt.trainer import Trainer
from tqdm import tqdm

def load_or_generate_dataset(dataset,dataset_directory,load_dataset = False):

    if load_dataset:
        try:
            true_dist, training_data = dataset.load_dataset(dataset_directory)
        except Exception as e:
            print("Exception occured while loading, generating dataset instead: " + e)
            temp = dataset.generate_all_possibilities()
            true_dist = dataset.generate_true_dist(alpha=alpha)
            training_data = dataset.sample_training_data(training_dataset_size, true_dist.tolist())
            dataset.save_dataset(dataset_directory)
    else:
        temp = dataset.generate_all_possibilities()
        true_dist = dataset.generate_true_dist(alpha=alpha)
        training_data = dataset.sample_training_data(training_dataset_size, true_dist.tolist())
        dataset.save_dataset(dataset_directory)
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

def train_model(model, train_data, trainer, model_directory,epochs):
  with open("training_logs.txt", "w") as log_file:
    log_message = f"Epoch, Loss\n"
    log_file.write(log_message)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        trainer.run()
        current_loss = trainer.loss.item()
        print(f"Training completed for epoch {epoch + 1}")
        log_message = f"{epoch + 1}, {current_loss:.5f}\n"
        log_file.write(log_message)

    torch.save(model.state_dict(), model_directory + "model.pt")

def batch_end_callback(trainer):
    global best_iter
    global best_epoch
    global model_filepath
    global training_log_filepath
    if trainer.iter_num == 0:
        with open(training_log_filepath, "w") as log_file:
            log_message = f"Iteration, Loss\n"
            log_file.write(log_message)
        return

    with open(training_log_filepath, "a") as log_file:
        log_message = f"{trainer.iter_num} {trainer.loss.item()}\n"
        log_file.write(log_message)

    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 100:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.loss.item() < best_iter:
            best_iter = trainer.loss.item()
            best_epoch = trainer.iter_num
            torch.save(model.state_dict(), model_filepath)

def save_run_config(run_config_filepath,run_number,seed,zipf_alpha,training_size,
                    true_dist_size,food_list_name,number_of_person, possibilities_size):
    run = {}
    run["run_number"] = run_number
    run["seed"] = seed
    run["zipf_alpha"] = zipf_alpha
    run["training_size"] = training_size
    run["true_dist_size"] = true_dist_size
    run["food_list_name"] = food_list_name
    run["number_of_person"] = number_of_person
    run["possibilities_size"] = possibilities_size
    json_str = json.dumps(run)
    with open(run_config_filepath, "a") as f:
        f.write(json_str + "\n")

def calculate_metrics(true_dist, training_data,collected_generations,results_filepath,run_number):
    # True dist
    true_dist_df = pd.DataFrame(true_dist, columns=["facts"])
    true_duplicates_count = true_dist_df.groupby(list(true_dist_df.columns)).size().reset_index(name='count_true')
    # Training_dist
    training_dist_df = pd.DataFrame(training_data, columns=["facts"])
    training_duplicates_count = training_dist_df.groupby(list(training_dist_df.columns)).size().reset_index(name='count_train')

    collected_generations_df = pd.DataFrame(collected_generations, columns=["facts"])
    collected_generations_counts = collected_generations_df.groupby(
        list(collected_generations_df.columns)).size().reset_index(name='count_generated')

    # Merge true dist and training dist dataframes, outer is used to include data that is not in training data as well
    merged_df = pd.merge(true_duplicates_count, training_duplicates_count, on='facts', how='outer')

    # Add generated_df to true and training dfs
    # outer can be used to include all facts in true dist
    # inner can be used to only show the comparison of generated facts
    comparison_df = pd.merge(merged_df, collected_generations_counts, on='facts', how='outer')

    # Fill in 0 for facts that not appear
    comparison_df = comparison_df.fillna(0)

    # Normalize the counts by length
    comparison_df["count_generated"] = comparison_df['count_generated'] / len(collected_generations)
    comparison_df["count_train"] = comparison_df['count_train'] / len(training_data)
    comparison_df["count_true"] = comparison_df['count_true'] / len(true_dist)

    comparison_df = comparison_df.sort_values(by=['count_generated'], ascending=False)

    # True hallucination rate (generations not in true dist)
    true_hallucinations = pd.merge(collected_generations_counts, true_duplicates_count, on='facts', how='left')
    true_hallucinations = true_hallucinations.fillna(0)
    try:
        number_of_true_hallucinations = true_hallucinations["count_true"].value_counts()[0]
        true_hallucinations_rate = number_of_true_hallucinations / len(collected_generations)
    except:
        number_of_true_hallucinations = 0
        true_hallucinations_rate = 0
    print(f"Rate of true hallucinations: {true_hallucinations_rate} ")

    true_hallucinations_df = true_hallucinations[true_hallucinations["count_true"] == 0]
    hallucinations_list = true_hallucinations_df["facts"].tolist()


    # Naive hallucination rate (every generation not in training data)
    naive_hallucinations = pd.merge(collected_generations_counts, training_duplicates_count, on='facts', how='left')

    naive_hallucinations = naive_hallucinations.fillna(0)
    try:
        number_of_naive_hallucinations = naive_hallucinations["count_train"].value_counts()[0]
        naive_hallucinations_rate = number_of_naive_hallucinations / len(collected_generations)
    except:
        number_of_naive_hallucinations = 0
        naive_hallucinations_rate = 0

    print(f"Rate of naive hallucinations: {naive_hallucinations_rate} ")

    try:
        MF = training_duplicates_count["count_train"].value_counts()[1] / len(training_data)
        mf_df = training_duplicates_count[training_duplicates_count["count_train"] < 4]
        monofact_list = mf_df["facts"].tolist()
    except:
        MF = 0
        monofact_list = []
    from src.lib.calibration import miscalibration
    comparison_sorted_by_generated = comparison_df.sort_values(by='count_generated', ascending=False)
    miscalibration_rate = miscalibration(comparison_sorted_by_generated['count_generated'],
                                         comparison_sorted_by_generated['count_true'])
    unique_names = len(set([t[1] for t in train_dataset]))
    unique_foods = len(set([t[2] for t in train_dataset]))
    # Possible generations
    POSS_GENERATIONS = unique_names * unique_foods

    # Facts to all possibilities - facts, approximated
    APPROX_FACTS_TO_POSSIBLE_HALLUCINATIONS = 300 * len(training_duplicates_count) / (
                POSS_GENERATIONS - len(training_duplicates_count))

    HALLUCINATION_RATE = true_hallucinations_rate

    # MF = 0.43875

    MISCALIBRATION = miscalibration_rate


    estimated_hallucination_rate = MF - MISCALIBRATION - (
                7 / np.sqrt(len(training_data))) - APPROX_FACTS_TO_POSSIBLE_HALLUCINATIONS

    experiment = {}
    experiment['run'] = run_number
    experiment['monofact_rate'] = MF
    experiment['miscalibration_rate'] = MISCALIBRATION
    experiment['facts_to_possible_hallucinations_ratio'] = APPROX_FACTS_TO_POSSIBLE_HALLUCINATIONS
    experiment['estimated_hallucinations_rate'] = estimated_hallucination_rate
    experiment['naive_hallucinations_rate'] = naive_hallucinations_rate
    experiment['true_hallucinations_rate'] = true_hallucinations_rate

    json_str = json.dumps(experiment)
    with open(results_filepath, 'a') as file:
        file.write(json_str + '\n')

    return hallucinations_list, monofact_list

def get_and_save_embeddings(model,train_dataset,tokenized_hallucinations,tokenized_monofacts,
                                        tokenized_generated_facts,embeddings_directory):
    final_hidden_states = []

    monofact_embeddings = []
    normal_fact_embeddings = []
    for fact in train_dataset:
        output, loss, hidden_states = model(fact.unsqueeze(0).long().to(device), output_hidden_states=True)
        fact = fact.detach().cpu().tolist()
        if fact in tokenized_monofacts:
            monofact_embeddings.append((fact, hidden_states[-1].flatten().detach()))
        else:
            normal_fact_embeddings.append((fact, hidden_states[-1].flatten().detach()))
        final_hidden_states.append((fact, hidden_states[-1].flatten().detach()))

    hallucination_embeddings = []
    for fact in tokenized_hallucinations:
        _, _, hidden_states = model(torch.tensor(fact).unsqueeze(0).long().to(device),
                                            output_hidden_states=True)

        hallucination_embeddings.append((fact, hidden_states[-1].flatten().detach()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ))

    generated_fact_embeddings = []
    for fact in tokenized_generated_facts:
        _, _, hidden_states = model(torch.tensor(fact).unsqueeze(0).long().to(device),
                                            output_hidden_states=True)

        generated_fact_embeddings.append((fact, hidden_states[-1].flatten().detach()))

    with open(os.path.join(embeddings_directory,"monofact_embeddings.json"), "w") as f:
        for line in monofact_embeddings:
            f.write(json.dumps((line[0], line[1].tolist())) + "\n")
    with open(os.path.join(embeddings_directory,"hallucination_embeddings.json"), "w") as f:
        for line in hallucination_embeddings:
            f.write(json.dumps((line[0], line[1].tolist())) + "\n")
    with open(os.path.join(embeddings_directory,"normal_fact_embeddings.json"), "w") as f:
        for line in normal_fact_embeddings:
            f.write(json.dumps((line[0], line[1].tolist())) + "\n")
    with open(os.path.join(embeddings_directory,"generated_fact_embeddings.json"), "w") as f:
        for line in generated_fact_embeddings:
            f.write(json.dumps((line[0], line[1].tolist())) + "\n")


seeds = [42, 15, 102]
alphas = [0.1, 0.5, 1.0, 1.5, 2]
training_sizes = [800,8000]
food_list_names = ["food_list_small.txt", "food_list_medium.txt"]
load_dataset = False
load_model = False
run = 0

model_directory = "./src/experiment/models"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print("Directory", model_directory, "created successfully.")

training_logs_directory = "./src/experiment/training_logs"
if not os.path.exists(training_logs_directory):
    os.makedirs(training_logs_directory)
    print("Directory", training_logs_directory, "created successfully.")

run_configs_directory = "./src/experiment/run_configs"
if not os.path.exists(run_configs_directory):
    os.makedirs(run_configs_directory)
    print("Directory", run_configs_directory, "created successfully.")
run_config_filepath = os.path.join(run_configs_directory, "run_configs.json")

results_directory = "./src/experiment/results"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
    print("Directory", results_directory, "created successfully.")
results_filepath = os.path.join(results_directory, "results.json")

for seed in tqdm(seeds):
    for alpha in tqdm(alphas):
        for food_list_name in tqdm(food_list_names):
            for training_size in tqdm(training_sizes):
                if training_size == 8000:
                    # Load dataset
                    true_dist_size = 10000
                    number_of_person = 1000
                else:
                    true_dist_size = 1000
                    number_of_person = 100
                training_dataset_size = int(0.8 * true_dist_size)
                alpha = alpha
                dataset = FactDatasetGenerator(number_person=1000, distribution="zipf", data_folder='src/data/',
                                   food_list_name=food_list_name, true_dist_size=true_dist_size,seed=seed)


                run_directory = f"./src/experiment/Runs/Run_{run}/"
                print(run_directory)
                if not os.path.exists(run_directory):
                    print(run_directory)
                    os.makedirs(run_directory)

                dataset_directory = os.path.join(run_directory, "dataset")
                if not os.path.exists(dataset_directory):
                    os.makedirs(dataset_directory)
                    print("Directory", dataset_directory, "created successfully.")


                true_dist, training_data = load_or_generate_dataset(dataset, dataset_directory=dataset_directory,
                                                                    load_dataset=load_dataset)

                save_run_config(run_config_filepath=run_config_filepath, run_number=run, seed=seed, zipf_alpha=alpha,
                                training_size=training_dataset_size, true_dist_size=true_dist_size,
                                food_list_name=food_list_name, number_of_person=number_of_person,
                                possibilities_size=len(dataset.all_possibilities))

                # Create datasets
                train_dataset = [torch.tensor(x, dtype=torch.long) for x in dataset.tokenized_training_data]
                train_data = create_dataset(train_dataset)

                # Model configuration
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = create_model(device, dataset)

                # Trainer configuration
                train_config = Trainer.get_default_config()
                train_config.learning_rate = 5e-5
                train_config.max_iters = 10000
                train_config.num_workers = 0
                # Create trainer object
                trainer = create_trainer(train_config, model, train_data)

                best_iter = 100000000000000
                best_epoch = 0

                model_filepath = os.path.join(model_directory, f"run_{run}_model.pt")
                training_log_filepath = os.path.join(training_logs_directory, f"run_{run}_training_log.txt")

                # Train the model
                trainer.set_callback('on_batch_end', batch_end_callback)

                if load_model:
                    model.load_state_dict(torch.load(model_filepath))
                else:
                    trainer.run()
                    print(f"Best loss is: {best_iter} on epoch: {best_epoch}")



                # Model evaluation
                model.eval()

                n_sequences = 1000
                from collections import defaultdict

                collected_generations = []

                for _ in range(n_sequences):
                    x = torch.Tensor([0]).unsqueeze(0).long().to(device)
                    y_gen = model.generate(x.detach(), 2, do_sample=True)
                    name = food_item = dataset.decode([y_gen[0][1]])[0]
                    food_item = dataset.decode([y_gen[0][2]])[0]
                    collected_generations.append(f"{name},{food_item}")

                hallucinations_list, monofact_list = calculate_metrics(true_dist=true_dist,training_data=training_data,
                                  collected_generations=collected_generations,
                                  results_filepath=results_filepath,run_number=run)

                tokenized_hallucinations = dataset.tokenize_data(hallucinations_list)
                tokenized_monofacts = dataset.tokenize_data(monofact_list)
                tokenized_generated_facts = dataset.tokenize_data(collected_generations)

                embeddings_directory = f"./src/experiment/Run_{run}/embeddings"
                if not os.path.exists(embeddings_directory):
                    os.makedirs(embeddings_directory)
                    print("Directory", embeddings_directory, "created successfully.")


                get_and_save_embeddings(model=model,train_dataset=train_dataset,
                                        tokenized_hallucinations=tokenized_hallucinations,
                                        tokenized_monofacts=tokenized_monofacts,
                                        tokenized_generated_facts=tokenized_generated_facts,
                                        embeddings_directory=embeddings_directory)

                run += 1





