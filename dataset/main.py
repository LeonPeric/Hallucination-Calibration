# names of persons are from: https://www.gutenberg.org/
# names of foods are from: http://www.ivl.disco.unimib.it/activities/food524db/
# names of small foods list is from food 101 of huggingface
from scipy import io
import numpy as np
import random


class DatasetGenerator:
    def __init__(
        self,
        dataset_folder="./data/",
        small=True,
        samples=10000,
        number_person=10,
        max_foods_per_person=10,
        distribution="zipf",
        place=False,
        day=False,
        seed=42,
        food_global=False,
    ):
        self.dataset_folder = dataset_folder
        self.food_list_small = small

        self.samples = samples
        self.number_person = number_person
        self.max_foods_per_person = max_foods_per_person
        self.distribution = distribution

        self.place = place
        self.day = day

        self.food_global = food_global

        self.rng = random.Random(x=seed)
        np.random.seed(seed)

        self.load_data()

    def calculate_zipf_distribution(self, items, alpha=1):
        temp = np.array([1 / ((i + 1) ** alpha) for i in range(len(items))])
        return temp / sum(temp)

    def create_list_with_probabilities(self, name, shuffle=True, alpha=1):
        with open(self.dataset_folder + name, "r", encoding="utf-8") as f:
            items = f.read().splitlines()
            if shuffle:
                self.rng.shuffle(items)

        if self.distribution == "zipf":
            probabilities = self.calculate_zipf_distribution(items, alpha=alpha)
        else:
            probabilities = np.array([1 for i in range(len(items))])

        return items, probabilities

    def load_data(self):
        # loading the names
        with open(self.dataset_folder + "names-f.txt", "r", encoding="utf-8") as f:
            names_f = f.read().splitlines()
        with open(self.dataset_folder + "names-m.txt", "r", encoding="utf-8") as f:
            names_m = f.read().splitlines()

        self.names = names_f + names_m

        # loading the food items and probabilies for each item
        if self.food_list_small:
            self.foods, self.foods_weighted = self.create_list_with_probabilities(
                "food_list_small.txt"
            )
        else:
            self.foods, self.foods_weighted = self.create_list_with_probabilities(
                "food_list_large.mat"
            )

        if self.place:
            self.restaurants, self.restaurants_weighted = (
                self.create_list_with_probabilities("restaurants.txt")
            )

        if self.day:
            self.days, self.days_weighted = self.create_list_with_probabilities(
                "days.txt"
            )

    def tokenize(self):
        assert self.dataset_splitted

        idx = 0
        self.word2id = {}
        self.dataset_tokenized = []

        for row in self.dataset_splitted:
            sent_tokenized = []
            for word in row:
                if word not in self.word2id:
                    self.word2id[word] = idx
                    idx += 1
                sent_tokenized.append(self.word2id[word])
            self.dataset_tokenized.append(sent_tokenized)

        self.vocabulary_size = idx

    def decode(self, tokenized):
        return [
            list(self.word2id.keys())[list(self.word2id.values()).index(i)]
            for i in tokenized
        ]

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def generate_probabilities(self):
        self.sampled_names = self.rng.sample(self.names, self.number_person)
        # Uniforn probabilities among names
        self.distribution_of_names = {}
        for name in self.sampled_names:
            self.distribution_of_names[name] = 1 / len(self.sampled_names)

        self.distributions_per_name = {}
        for name in self.sampled_names:
            food_names, food_dist = self.create_list_with_probabilities(
                "food_list_medium.txt", alpha=2
            )
            # normalize food_dist so it sums to 1
            self.distributions_per_name[name] = {
                "food": food_names[: self.max_foods_per_person],
                "prob": food_dist[: self.max_foods_per_person]
                / np.sum(food_dist[: self.max_foods_per_person]),
            }

    def generate(self, one_token_per_attr=True):
        dataset = []
        dataset_spliited = []

        template_no_place_no_day = "{name} {food}"
        template_place_no_day = "{name} {food} {place}"
        template_no_place_day = "{name} {food} {day}"
        template_place_day = "{name} {food} {place} {day}"

        while len(dataset) < self.samples:
            if self.food_global:
                name = np.random.choice(
                    list(self.distribution_of_names.keys()),
                    p=list(self.distribution_of_names.values()),
                    size=1,
                )[0]

                food = np.random.choice(self.foods, p=self.foods_weighted, size=1)[0]

            else:
                # now generate the foods which that person likes
                name = np.random.choice(
                    list(self.distribution_of_names.keys()),
                    p=list(self.distribution_of_names.values()),
                    size=1,
                )[0]

                food = np.random.choice(
                    self.distributions_per_name[name]["food"],
                    p=self.distributions_per_name[name]["prob"],
                    size=1,
                )[0]

            if self.place:
                random_place = np.random.choice(self.restaurants, 1)[0] + " "
            else:
                random_place = ""
            if self.day:
                random_day = np.random.choice(self.days, 1)[0]
            else:
                random_day = ""

            if self.place and self.day:
                template = template_place_day
            elif self.place:
                template = template_place_no_day
            elif self.day:
                template = template_no_place_day
            else:
                template = template_no_place_no_day

            fact = template.format(
                name=name,
                food=food.strip(),
                place=random_place.strip(),
                day=random_day.strip(),
            )

            dataset.append(fact)

            if one_token_per_attr:
                fact_spliited = [
                    elem
                    for elem in [
                        name,
                        food.strip(),
                        random_place.strip(),
                        random_day.strip(),
                    ]
                    if len(elem)
                ]
                dataset_spliited.append(fact_spliited)
            else:
                dataset_spliited.append(fact.split())

        self.dataset = dataset
        self.dataset_splitted = dataset_spliited

        with open(self.dataset_folder + "dataset.txt", "w", encoding="utf-8") as f:
            for line in dataset:
                f.write(line + "\n")

    ## split the tokenized data into train and test, also shuffle the data
    def split(self, train_ratio=0.8):
        assert self.dataset_tokenized

        np.random.shuffle(self.dataset_tokenized)
        train_size = int(len(self.dataset_tokenized) * train_ratio)
        self.train = self.dataset_tokenized[:train_size]
        self.test = self.dataset_tokenized[train_size:]
