from __future__ import unicode_literals, print_function, division

import os
import torch
import math

import utils
from utils import Data
import random


class NamesNationalityDataset:
    def __init__(self, device, _nations, _path, _x_train, _x_test):
        self.device = device
        self.categories = _nations
        self.categories_data = {}

        # Read a file and split into lines
        self.dataset = list()
        self.x_train = _x_train
        self.x_test = _x_test
        for cat in self.categories:
            filename = str(_path) + cat + '.txt'
            category = os.path.splitext(os.path.basename(filename))[0]
            lines = utils.read_file_lines(filename)
            self.categories_data[category] = lines

        self.n_categories = len(self.categories)
        self.dateset_produce()
        self.data = {"train": (self.dataset_split(self.x_train, self.x_test))[0],
                     "test": (self.dataset_split(self.x_train, self.x_test))[1],
                     "valid": (self.dataset_split(self.x_train, self.x_test))[2]}

    def category_of_output(self, output):
        top_n, top_i = output.topk(1)
        category_inx = top_i[0].item()
        return self.categories[category_inx], category_inx

    def dataset_padd(self):
        max_len = -1
        for d in self.dataset:
            max_len = max(max_len, d.get_xt().shape[0])
        print(max_len)

    def dateset_produce(self):

        for _cat in self.categories:
            for word in self.categories_data[_cat]:
                self.dataset.append(Data(word, _cat, torch.tensor([self.categories.index(_cat)])))

    def dataset_split(self, x_train, x_test):
        random.shuffle(self.dataset)
        size_ds = len(self.dataset)
        size_train = math.ceil(size_ds * x_train)
        size_test = math.ceil(size_ds * x_test)

        __train = self.dataset[0:size_train]
        __test = self.dataset[size_train:size_train + size_test]
        __valid = self.dataset[size_train + size_test:]
        return __train, __test, __valid

    def random_data(self, dataset):
        data = utils.random_choice(dataset)
        return data
