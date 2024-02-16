import math
import torch

###############################################################################


class ChooseLR():

    def __init__(
        self, optimizer, start_lr=1.0, end_lr=1e-10, default_lr=0.1,
        factor=0.05, threshold=0.0
    ):

        self.__start_lr = start_lr
        self.__end_lr = end_lr
        self.__default_lr = default_lr
        self.__tmp_lr = start_lr

        if(factor >= 1.0):
            raise ValueError("Factor should be < 1.0")
        self.__factor = factor

        if(not isinstance(optimizer, torch.optim.Optimizer)):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.__optim = optimizer

        self.__threshold = threshold
        self.__metrics_init = None
        self.__params_init = None
        self.__chosen = False

    def __save_params(self):
        self.__params_init = []
        for group in self.__optim.param_groups:
            self.__params_init.append([])
            for w in group["params"]:
                self.__params_init[-1].append(w.data)

    def __load_params(self):
        for i, group in enumerate(self.__optim.param_groups):
            for j, w in enumerate(group["params"]):
                w.data = self.__params_init[i][j]

    def step(self, metrics=None):

        # If the learning rate is chosen, we can exit
        if(self.__chosen):
            return -1

        # If the learning rate is too small, then we get the default learning
        # rate
        if(self.__tmp_lr < self.__end_lr):
            for group in self.__optim.param_groups:
                self.__load_params()
                group["lr"] = self.__default_lr
            del self.__params_init
            self.__chosen = True
            return 1

        # If metrics_init is None, we need to check the metrics with lr=0
        if(self.__metrics_init is None and metrics is None):
            for group in self.__optim.param_groups:
                group["lr"] = 0.0
                self.__save_params()
            return 0

        # If metrics_init is None and we have a metric, we save it
        if(self.__metrics_init is None):
            self.__metrics_init = metrics
            for group in self.__optim.param_groups:
                group["lr"] = self.__start_lr
                self.__tmp_lr = self.__start_lr
            return 0

        # If the metric is higher than the metric at initialization, we
        # decrease the learning rate
        if(math.isnan(metrics) or math.isinf(metrics)
           or metrics-self.__metrics_init > self.__threshold
           ):
            for group in self.__optim.param_groups:
                self.__load_params()
                group["lr"] = group["lr"]*self.__factor
                self.__tmp_lr = self.__tmp_lr*self.__factor
            return 0

        # Otherwise, we have chosen the right learning rate!
        del self.__params_init
        self.__chosen = True
        return 1
