import math
import numpy as np
import random
import torch
from module.model import Model


class MNISTModel(Model):

    def __init__(self, **kwargs):

        if("seed" in kwargs):
            super().__init__(seed=kwargs["seed"])
        else:
            super().__init__()

        self.input_size = None
        if("input_size" in kwargs):
            self.input_size = kwargs["input_size"]

        # NOTE: Assuming 10 classes
        # Iniatializing the parameters (mean and variance of gaussian)
        self.__weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(10, 1, 5, 5)),
            torch.nn.Parameter(torch.empty(10, 10, 5, 5)),
            torch.nn.Parameter(torch.empty(10, 10, 5, 5)),
            torch.nn.Parameter(torch.empty(10, 10, 5, 5)),
        ])
        self.__bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(10)),
            torch.nn.Parameter(torch.empty(10)),
            torch.nn.Parameter(torch.empty(10)),
            torch.nn.Parameter(torch.empty(10)),
        ])

        # Initializing stride and padding
        self.__param_list = [[1, 1], [2, 1], [1, 1], [1, 1]]

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initializing weights with Xavier initializer
        for i in range(len(self.__weight)):
            torch.nn.init.xavier_uniform_(self.__weight[i])
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.__weight[i])
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(
                self.__bias[i], -bound, bound)

    def forward(self, batch):

        x = batch["x"]

        if(self.input_size is None):
            self.input_size = list(x.shape[1:])

        # Forwarding in the layers
        for i in range(len(self.__weight)):

            # Forwarding in convolution and activation
            x = torch.nn.functional.conv2d(
                x, (self.__weight[i]),
                bias=(self.__bias[i]),
                stride=self.__param_list[i][0],
                padding=self.__param_list[i][1]
            )
            x = torch.nn.functional.leaky_relu(x)

        # Forwarding average pooling
        x = torch.nn.functional.avg_pool2d(x, 8)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)

        return x
