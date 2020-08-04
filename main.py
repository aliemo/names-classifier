from dataset import NamesNationalityDataset
import torch.nn as nn
import utils
import model
import networks
import torch


class Configs:
    def __init__(self):
        self.nations = ["Iranian", "English", "Italian", "Japanese", "Czech", "Arabic"]
        self.raw_data_path = 'data/names/'
        self.data_split = {"train": .7, "test": .2}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.NLLLoss()
        self.learning_rate = 0.001  # If you set this too high, it might explode. If too low, it might not learn
        self.optimizer = torch.optim.Adam
        self.n_hidden = 64
        self.n_layers = 1
        self.n_batch = 1
        self.n_max_len = 16
        self.n_inputs = utils.n_letters
        self.n_outputs = len(self.nations)
        self.x_train = .8
        self.x_test = .2

        self.confusion = torch.zeros(self.n_outputs, self.n_outputs)
        self.n_test = 100000


class LearningModel:
    def __init__(self):

        self.train_iterations = 25000
        self.train_log_cycle = 5000
        self.train_plot_cycle = 1000
        self.test_iterations = 10000
        self.test_log_cycle = 1000


lm = LearningModel()
conf = Configs()


ds = NamesNationalityDataset(conf.device, conf.nations, conf.raw_data_path, conf.x_train, conf.x_test)
network = networks.LSTM(conf.n_inputs, conf.n_hidden, conf.n_outputs, conf.n_batch, conf.n_layers, conf.device)
conf.optimizer = torch.optim.Adam(network.parameters(), lr=conf.learning_rate)
model = model.RecurrentModel(network, ds, conf.criterion, conf.optimizer, lm, conf.confusion, conf.device)
model.train()
model.test()
model.plot_lost()
model.show_accuracy_matrix()
model.calculate_accuracy()
print(model.accuracy)
