from __future__ import unicode_literals, print_function, division

import time

import numpy

import utils
import dataset
import matplotlib.pyplot as plt


class RecurrentModel:

    def __init__(self, model, ds: dataset.NamesNationalityDataset, criterion, optimizer, lm, confusion, device):
        self.model = model
        self.ds = ds
        self.loss = 0
        self.criterion = criterion
        self.optimizer = optimizer
        self.losses = []
        self.lm = lm
        self.device = device
        self.current_loss = 0
        self.confusion = confusion
        self.acc_list = []
        self.accuracy = -1

    def train_step(self, _x, _y):
        self.model.zero_grad()
        self.model.hidden = self.model.init_hidden()
        output = self.model(_x)[-1]
        loss = self.criterion(output.unsqueeze(0), _y)
        loss.backward()
        self.optimizer.step()
        return output.unsqueeze(0), loss.item()

    def train(self):
        start = time.time()
        for itr in range(1, self.lm.train_iterations + 1):
            data = self.ds.data["train"][itr % len(self.ds.data["train"])]
            # print(line_tensor)
            output, loss = self.train_step(data.get_xt().to(self.device), data.get_yt().to(self.device))
            self.current_loss += loss

            # Print iter number, loss, name and guess
            if itr % self.lm.train_log_cycle == 0:
                guess, guess_i = self.ds.category_of_output(output)
                correctness = '✓' if guess == data.get_y() else '✗ (%s)' % data.get_y()
                print(
                    '%d %d%% (%s) %.4f %s / %s %s' % (
                        itr, itr / self.lm.train_iterations * 100, utils.time_since(start), loss, data.get_x(), guess,
                        correctness))

            # Add current loss avg to list of losses
            if itr % self.lm.train_plot_cycle == 0:
                self.losses.append(self.current_loss / self.lm.train_plot_cycle)
                self.current_loss = 0

    def eval(self, _x):
        self.model.hidden = self.model.init_hidden()
        out = self.model(_x)
        return out

    def test(self):

        for i in range(self.lm.test_iterations):

            data = self.ds.data["test"][i % len(self.ds.data["test"])]
            output = self.eval(data.get_xt().to(self.device))
            output = output.unsqueeze(0)
            guess, guess_i = self.ds.category_of_output(output)
            category_i = self.ds.categories.index(data.get_y())
            self.confusion[category_i][guess_i] += 1
            if self.lm.test_log_cycle == -1:
                if i % self.lm.test_iterations / 20 == 0:
                    print(f"Progress {round(i / self.lm.test_iterations * 100, 2)}%")

            else:
                if i % self.lm.test_log_cycle == 0:
                    print(f"Progress is {i}/{self.lm.test_iterations} ({round(i / self.lm.test_iterations * 100, 2)}%)")

    def plot_lost(self):
        plt.figure()
        plt.plot(self.losses)
        plt.show()

    def show_accuracy_matrix(self):
        for i in range(len(self.ds.categories)):
            self.confusion[i] = self.confusion[i] / self.confusion[i].sum()
        utils.confusion_plot(self.confusion, self.ds.categories)

    def calculate_accuracy(self):
        for i in range(len(self.ds.categories)):
            self.acc_list.append(self.confusion[i][i])

        self.accuracy = numpy.sum(self.acc_list) / self.ds.n_categories
