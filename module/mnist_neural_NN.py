import math
import numpy as np
import random
import torch
from module.model import Model


class MNISTComplexityMeasure(Model):

    def __init__(self, **kwargs):

        if("seed" in kwargs):
            super().__init__(seed=kwargs["seed"])
        else:
            super().__init__()

        self.__hidden_size = 1024

        self.__param_size = 10*1*5*5
        self.__param_size += 3*(10*10*5*5)
        self.__param_size += 4*(10)

        # Iniatializing the parameters
        self.__weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(
                self.__hidden_size, self.__param_size)),
            torch.nn.Parameter(torch.empty(
                self.__hidden_size, self.__hidden_size)),
            torch.nn.Parameter(torch.empty(
                self.__hidden_size, self.__hidden_size)),
            torch.nn.Parameter(torch.empty(1, self.__hidden_size)),
        ])
        self.__bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(self.__hidden_size)),
            torch.nn.Parameter(torch.empty(self.__hidden_size)),
            torch.nn.Parameter(torch.empty(self.__hidden_size)),
            torch.nn.Parameter(torch.empty(1)),
        ])

        # We initialize the weights for the layernorm layers
        self.__norm_weight = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.__param_size)),
            torch.nn.Parameter(torch.ones(self.__hidden_size)),
            torch.nn.Parameter(torch.ones(self.__hidden_size)),
            torch.nn.Parameter(torch.ones(self.__hidden_size)),
        ])
        self.__norm_bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(self.__param_size)),
            torch.nn.Parameter(torch.zeros(self.__hidden_size)),
            torch.nn.Parameter(torch.zeros(self.__hidden_size)),
            torch.nn.Parameter(torch.zeros(self.__hidden_size)),
        ])
        self.__norm_mean = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(self.__param_size),
                               requires_grad=False),
        ])
        self.__norm_var = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.__param_size),
                               requires_grad=False),
        ])

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

        x = None
        for key in batch.keys():
            if(key not in ["y", "norm", "step"]):
                if(x is None):
                    x = torch.flatten(batch[key], start_dim=1)
                else:
                    x = torch.concat(
                        (x, torch.flatten(batch[key], start_dim=1)), dim=1)

        step = batch["step"][0]
        if(step == "train" and x.shape[0] > 1):
            training = True
        else:
            training = False

        x = torch.nn.functional.batch_norm(
            x, self.__norm_mean[0], self.__norm_var[0],
            weight=self.__norm_weight[0], bias=self.__norm_bias[0],
            training=training, momentum=0.1, eps=0.0)

        # Forwarding in the layers
        for i in range(len(self.__weight)):

            # Forwarding in convolution and activation
            x = torch.nn.functional.linear(
                x, self.__weight[i],
                bias=self.__bias[i],
            )
            if(i < len(self.__weight)-1):
                x = torch.nn.functional.leaky_relu(x)

        x = x**2.0

        return x
