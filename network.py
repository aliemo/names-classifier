import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softMax = nn.LogSoftmax(dim=1)

    def forward(self, _input, _hidden):
        combined = torch.cat((_input, _hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softMax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
