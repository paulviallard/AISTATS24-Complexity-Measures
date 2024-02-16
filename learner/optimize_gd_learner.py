import torch
import numpy as np
import logging
import random
from sklearn.base import BaseEstimator, ClassifierMixin


###############################################################################

class OptimizeGDLearner(BaseEstimator, ClassifierMixin):

    def __init__(
        self, batch_size=None, sampler=None, shuffle=False, seed=0
    ):
        self.data = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.sampler = sampler

    def fit(self, data=None):

        self.data = data

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        if(self.data is None):
            raise RuntimeError("self.data must be defined")
        if(not(isinstance(self.data, torch.utils.data.Dataset))):
            raise RuntimeError("self.data must be torch.utils.data.Dataset")

        if(self.batch_size is None):
            self.batch_size = len(self.data)
        self.loader = torch.utils.data.DataLoader(
            self.data, batch_size=self.batch_size,
            sampler=self.sampler, shuffle=self.shuffle)

        # Computing batch size
        num_batch = int(len(self.data)/self.batch_size)
        if(len(self.data) % self.batch_size != 0):
            num_batch += 1

        self._begin()

        while(not(self._meet_condition())):

            self._begin_epoch()

            for self._i, self._batch in enumerate(self.loader):

                self._begin_batch()

                # Optimize the model
                # REQUIRED: the dict self._log
                self._log = {}
                self._optimize()

                # Printing loss
                logging_str = "[{}/{}]".format(
                        self._i+1, num_batch)
                for key, value in self._log.items():
                    logging_str += self.__print_logging(key, value)
                logging.info(logging_str+"\r")

                if self._i+1 == num_batch:
                    logging.info("\n")

                self._end_batch()

            self._end_epoch()

        return self._end()

    def _meet_condition(self):
        raise NotImplementedError

    def _begin(self):
        pass

    def _end(self):
        return None

    def _begin_epoch(self):
        pass

    def _end_epoch(self):
        pass

    def _begin_batch(self):
        pass

    def _end_batch(self):
        pass

    def _optimize(self):
        raise NotImplementedError

    def __print_logging(self, key, value):
        if(isinstance(value, int)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, float)):
            return " - {} {:.4f}".format(key, value)
        elif(isinstance(value, str)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, torch.Tensor)):
            return self.__print_logging(key, value.cpu().detach().numpy())
        elif(isinstance(value, np.ndarray)):
            if(value.ndim == 0):
                return self.__print_logging(key, value.item())
            else:
                raise ValueError("value cannot be an array")
        else:
            raise TypeError(
                "value must be of type torch.Tensor; np.ndarray;"
                + " int; float or str.")

    def save(self):
        return self.model.state_dict()

    def load(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def numpy_to_torch(self, *var_list):
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(var_list[i], np.ndarray)):
                new_var_list.append(
                    torch.tensor(var_list[i], device=self.model.device))
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

    def torch_to_numpy(self, ref, *var_list):
        # Note: elements in var_list are considered as tensor
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(ref, np.ndarray)
               and isinstance(var_list[i], torch.Tensor)):
                new_var_list.append(var_list[i].detach().cpu().numpy())
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

###############################################################################
