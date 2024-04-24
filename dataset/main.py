# names of persons are from: https://www.gutenberg.org/
# names of foods are from: http://www.ivl.disco.unimib.it/activities/food524db/
# names of small foods list is from food 101 of huggingface
from scipy import io
import numpy as np


class Dataset:
    def __init__(
        self,
        dataset_folder="./data/",
        small=True,
        samples=10000,
        facts_per_person=10,
        distribution="zipf",
        place=False,
        time=False,
        day=False,
    ):
        self.dataset_folder = dataset_folder
        self.food_list_small = small

        self.samples = samples
        self.facts_per_person = facts_per_person
        self.distribution = distribution

        self.place = place
        self.time = time
        self.day = day

        self.load_data()

    def calculate_zipf_distribution(self, items):
        temp = np.array([1 / (i + 1) for i in range(len(items))])
        return temp / sum(temp)

    def create_list_with_probabilities(self, name):
        with open(self.dataset_folder + name, "r", encoding="utf-8") as f:
            items = f.read().splitlines()

        if self.distribution == "zipf":
            probabilities = self.calculate_zipf_distribution(items)
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

    def generate(self):
        dataset = []
        # first generate the amount of names required
        names = np.random.choice(
            self.names, self.samples // self.facts_per_person, replace=False
        )

        for name in names:
            # now generate the foods which that person likes
            foods = np.random.choice(
                self.foods, self.facts_per_person, replace=False, p=self.foods_weighted
            )

            if self.place:
                place = "at " + np.random.choice(self.restaurants, 1)[0] + " "
            else:
                place = ""

            if self.day:
                day = np.random.choice(self.days, 1)[0]
            else:
                day = ""

            for food in foods:
                fact = f"{name} had "
                fact = fact + food + place + day

                dataset.append(fact)

        self.dataset = dataset

        with open(self.dataset_folder + "dataset.txt", "w", encoding="utf-8") as f:
            for line in dataset:
                f.write(line + "\n")
