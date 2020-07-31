from __future__ import division, print_function, unicode_literals

import os
import pandas as pd
import glob
from io import open
import unicodedata
import string
import torch


class NamesClassifyDB:
    def __init__(self):
        self.names_data = {}
        self.names_categories = []
        self.letters = string.ascii_letters + ".,;"
        self.numberOfLetters = len(self.letters)

    def unicode_to_ascii(self, st):
        return ''.join(
            c for c in unicodedata.normalize('NFD', st)
            if unicodedata.category(c) != 'Mn'
            and c in self.letters
        )

    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicode_to_ascii(line) for line in lines]

    def letter_to_index(self, letter):
        return self.letters.find(letter)

    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.numberOfLetters)
        tensor[0][self.letter_to_tensor(letter)] = 1
        return tensor

    def line_to_tensor(self, datalist):
        tensor = torch.zeros(len(datalist), 1, self.numberOfLetters)
        for li, letter in enumerate(datalist):
            tensor[li][0][self.line_to_tensor(letter)] = 1
        return tensor

    @staticmethod
    def find_files(path='.'):
        return glob.glob(path)

    def update_from_texts_path(self, path='data/names/txt/*'):
        # try:
        for filename in self.find_files(path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.names_categories.append(category)
            __lines = self.read_lines(filename)
            self.names_data[category] = __lines

    def cats(self):
        return self.names_categories

    def data_size(self, category):
        if self.names_categories.__contains__(category):
            return len(self.names_data[category])
        else:
            print("ERROR: Database not contains %s category." % category)

    def data(self, category):
        if self.names_categories.__contains__(category):
            return self.names_data[category]
        else:
            print("ERROR: Database not contains %s category." % category)

    def size(self):
        return len(self.names_categories)

    def names_size(self, category='ignore'):
        if category == 'ignore':
            return len(self.names_data[self.names_categories[0]])
        if category in self.names_categories:
            return len(self.names_data[category])
        else:
            print("ERROR: Database not contains %s category." % category)
            return -1

    def save_to_csv(self, category, pathname='ignore'):
        df = pd.DataFrame({category: self.names_data[category]})
        if category == 'ignore':
            df.to_csv('./' + category + '.csv')
        else:
            df.to_csv(pathname + '/' + category + '.csv')

    def test(self, _cat='English', txt_path='data/names/txt', csv_path='data/names/csv'):
        self.update_from_texts_path((txt_path + '/*'))
        print(self.data(_cat))
        self.save_to_csv(_cat, pathname=csv_path)
