import torch
from module.module import Module

###############################################################################


class MeanStd(Module):

    def forward(self, x):
        ref = x
        x = self.to_torch(x)
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        return self.from_torch(ref, x_mean, x_std)


class StandardScaler(Module):

    def __init__(self):
        super().__init__()
        self.__mean_std = Module("MeanStd")

    def forward(self, x, x_mean=None, x_std=None):
        ref = x
        x = self.to_torch(x)
        if(x_mean is None or x_std is None):
            x_mean_, x_std_ = self.__mean_std(x)
            if(x_mean is None):
                x_mean = x_mean_
            if(x_std is None):
                x_std = x_std_

        x = (x-x_mean)/x_std
        return self.from_torch(ref, x)
