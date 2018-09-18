from abc import abstractmethod
import abc

import numpy as np
from torch import optim

LOSS_MEAN_WINDOW = 1000
PRINT_LOSS_MEAN_ITERATION = 100


class BaseTrainer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, learning_rate, model):
        self.iteration = 0
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.running_loss = np.ones(LOSS_MEAN_WINDOW)
        self.running_loss_idx = 0
        self.criterion = self.get_criterion()
        print(self.model)

    @abstractmethod
    def get_criterion(self):
        # Should return torch criterion
        pass

    @abstractmethod
    def get_loss(self, batch_data):
        # Should return torch loss object. 
        pass

    def train(self, batch_data):
        self.optimizer.zero_grad()
        loss = self.get_loss(batch_data)
        loss.backward()
        self.running_loss[self.running_loss_idx] = loss.data[0]
        if self.running_loss_idx >= LOSS_MEAN_WINDOW - 1:
            self.running_loss_idx = 0
        else:
            self.running_loss_idx += 1
        mean_loss = np.sum(self.running_loss) / LOSS_MEAN_WINDOW
        if (self.iteration % PRINT_LOSS_MEAN_ITERATION) == 0:
            print("loss: {}".format(mean_loss))
        self.optimizer.step()
        self.iteration += 1
