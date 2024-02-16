import torch

###############################################################################


class LambdaLR():

    def __init__(self, optimizer, decay=0.0):

        if(not isinstance(optimizer, torch.optim.Optimizer)):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.__optim = optimizer

        self.__decay = decay
        self.__lr_init_list = []

    def step(self, epoch):

        # If the learning rates was not saved, we save them
        if(len(self.__lr_init_list) == 0):
            for group in self.__optim.param_groups:
                self.__lr_init_list.append(group["lr"])

        # We compute the new learning rates
        for i, group in enumerate(self.__optim.param_groups):
            group["lr"] = self.__lr_init_list[i]*(epoch+1.0)**(-self.__decay)
