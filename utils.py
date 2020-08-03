from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch
import random
import time
import math

import matplotlib.pyplot as plt
from matplotlib import ticker

letters = string.ascii_letters + " .,;'"
n_letters = len(letters)


def files_in_path(path):
    return glob.glob(path)


def random_choice(lst):
    return lst[random.randint(0, len(lst) - 1)]


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in letters
    )


def read_file_lines(file):
    _lines = open(file, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(_) for _ in _lines]


def letter_indexer(letter):
    return letters.find(letter)


def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_indexer(letter)] = 1
    return tensor


def category_of_output(out, categories):
    top_n, top_i = out.topk(1)
    category_i = top_i[0].item()
    return categories[category_i], category_i


def word_to_tensor(word, pad_size=0):

    tensor = torch.zeros(len(word) + pad_size, 1, n_letters)
    for li, letter in enumerate(word):
        tensor[li][0][letter_indexer(letter)] = 1
    return tensor


class Data:

    def __init__(self, x, y, yt):
        self.x = x
        self.y = y
        self.xt = word_to_tensor(self.x)
        self.yt = yt

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_xt(self):
        return self.xt

    def get_yt(self):
        return self.yt

    def __str__(self):
        return f"(x: {self.get_x()}, y: {self.get_y()})"


def confusion_plot(matrix, y_category):
    """
    A function that plots a confusion matrix

    :param matrix: Confusion matrix
    :param y_category: Names of categories.
    :return: NA
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y_category, rotation=90)
    ax.set_yticklabels([''] + y_category)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
