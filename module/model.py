import torch
from module.module import Module


class Model(Module):

    def __init__(self, seed=0):
        super().__init__()
        self.device = None
        self.seed = seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def to(self, device=None):

        if(device is None):
            return
        elif(isinstance(device, str) or isinstance(device, torch.device)):
            self.__to(device)
        elif(isinstance(device, list)):
            for i in range(len(device)):
                try:
                    self.__to(device[i])
                    return
                except AssertionError:
                    pass
                except RuntimeError:
                    pass
        else:
            raise ValueError(
                "device must be either str, torch.device or a list")

    def __to(self, device=None):

        if(isinstance(device, str)):
            self.device = torch.device(device)
        elif(isinstance(device, torch.device)):
            self.device = device
        else:
            raise ValueError(
                "device in the list must be either str or torch.device")

        super().to(self.device)
