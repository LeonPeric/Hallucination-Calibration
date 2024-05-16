# names of persons are from: https://www.gutenberg.org/
# names of foods are from: http://www.ivl.disco.unimib.it/activities/food524db/
# names of small foods list is from food 101 of huggingface
import os

from scipy import io
import numpy as np
import random


class FactDatasetGenerator:
    def __init__(
        self,
        dataset_folder="./fact_data/",
        number_person=10,
        distribution="zipf",
        seed=42,
        food_list_name="food_list_small.txt",
        true_dist_size=1000,
        experiment_path="./experiment/data/",
    ):
        self.dataset_folder = dataset_folder
        self.experiment_path = experiment_path

        self.food_list_name = food_list_name
        self.true_dist_size = true_dist_size

        self.number_person = number_person

        self.distribution = distribution

        self.rng = random.Random(x=seed)
        np.random.seed(seed)

    def generate_all_possibilities(self):
        possible_facts = []

        # Read female names
        with open(self.dataset_folder + "names-f.txt", "r", encoding="utf-8") as f:
            names_f = f.read().splitlines()
        # Read male names
        with open(self.dataset_folder + "names-m.txt", "r", encoding="utf-8") as f:
            names_m = f.read().splitlines()

        # Combine names
        self.all_names = names_f + names_m

        # Sample required number of names
        self.names = self.rng.sample(self.all_names, self.number_person)

        # Read food names
        with open(
            self.dataset_folder + self.food_list_name, "r", encoding="utf-8"
        ) as f:
            self.foods = f.read().splitlines()

        # Create all possible facts   -- Y set in our drawing
        for i in range(len(self.names)):
            for j in range(len(self.foods)):
                possible_facts.append(self.names[i] + "," + self.foods[j])

        self.rng.shuffle(possible_facts)

        self.all_possibilities = possible_facts

        self.save_file("all_facts.txt", self.all_possibilities)

        return possible_facts

    def calculate_zipf_distribution(self, items, alpha=1):
        temp = np.array([1 / ((i + 1) ** alpha) for i in range(len(items))])
        return temp / sum(temp)

    def sample_zipf_distribution(self, alpha, possibilities, size=1):
        # Sample from the Zipf distribution
        probabilities = self.calculate_zipf_distribution(possibilities, alpha)

        samples = np.random.choice(possibilities, size=size, p=probabilities)
        return samples

    def generate_true_dist(self):

        # Generates true distribution by sampling from all possible facts Y (blue + green in the drawing)
        self.true_dist = self.sample_zipf_distribution(
            alpha=1, possibilities=self.all_possibilities, size=self.true_dist_size
        )
        self.tokenize_true_dist()

        return self.true_dist

    def tokenize(self, words):

        # Tokenize the given words
        self.word2id["<s>"] = 0
        tokenized_words = []

        for word in words:
            # if word not seen before
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)

            tokenized_words.append(self.word2id[word])
        return tokenized_words

    def decode(self, tokenized):
        return [
            list(self.word2id.keys())[list(self.word2id.values()).index(i)]
            for i in tokenized
        ]

    def sample_training_data(self, n, data):
        # Samples the training data uniformly from the true distribution (greens in the drawing)
        self.training_data = self.sample_data(n, data)
        self.tokenized_training_data = self.tokenize_data(self.training_data)

        self.save_file("training_data.txt", self.training_data)

        return self.training_data

    def sample_data(self, n, data):

        sampled_data = self.rng.sample(data, n)

        return sampled_data

    def tokenize_true_dist(self):
        # tokenize the true dist at the beginning to create vocabulary. Training set might not include all tokens
        self.word2id = {}
        self.tokenized_true_dist = self.tokenize_data(self.true_dist)
        self.vocab_size = len(self.word2id)

    def tokenize_data(self, data):
        # tokenize given data, also adds start token
        tokenized_data = []
        for line in data:
            words = ["<s>"]
            words.extend(line.split(","))

            sent_tokenized = self.tokenize(words)

            tokenized_data.append(sent_tokenized)
        return tokenized_data

    def save_file(self, file_name, data):
        with open(self.experiment_path + file_name, "w") as f:
            for line in data:
                f.write(line + "\n")

    def load_file(self, file_name):
        with open(self.experiment_path + file_name, "r") as f:
            return f.read().splitlines()

    def load_dataset(self):
        # first load all the possible facts
        self.all_possibilities = self.load_file("all_facts.txt")

        # generate the true distribution
        self.generate_true_dist()

        # load the training data
        self.training_data = self.load_file("training_data.txt")
        self.tokenized_training_data = self.tokenize_data(self.training_data)
