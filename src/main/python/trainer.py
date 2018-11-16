import abc
from abc import abstractmethod

import numpy as np
from torch import optim
import torch.distributed as dist

LOSS_WINDOW_SIZE = 1000
PRINT_LOSS_MEAN_INTERVAL = 100


class BaseTrainer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, learning_rate, nn_module, world_size):
        self.iteration = 0
        self.nn_module = nn_module
        self.optimizer = optim.Adam(self.nn_module.parameters(), lr=learning_rate)
        self.loss_window = np.ones(LOSS_WINDOW_SIZE)
        self.loss_window_idx = 0
        self.criterion = self.get_criterion()
        self.loss_window_full = False
        self.world_size = world_size
        print(self.nn_module)

    @abstractmethod
    def get_criterion(self):
        # Should return torch criterion
        pass

    @abstractmethod
    def get_loss(self, batch_data):
        # Should return torch loss object.
        pass

    def average_gradients(self):
        for param in self.nn_module.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
            param.grad.data /= self.world_size

    def train(self, batch_data):
        self.optimizer.zero_grad()
        loss = self.get_loss(batch_data)
        loss.backward()
        self.loss_window[self.loss_window_idx] = loss.data
        if self.loss_window_idx < LOSS_WINDOW_SIZE - 1:
            self.loss_window_idx += 1
        else:
            self.loss_window_idx = 0
            self.loss_window_full = True
        if self.loss_window_full:
            mean_loss = np.mean(self.loss_window)
        else:
            mean_loss = np.mean(self.loss_window[: self.loss_window_idx])
        if (self.iteration % PRINT_LOSS_MEAN_INTERVAL) == 0:
            print("loss: {}".format(mean_loss))
        # Only average gradients across workers if there is more than 1.
        if self.world_size > 1:
            self.average_gradients()
        self.optimizer.step()
        self.iteration += 1
        return loss.data
