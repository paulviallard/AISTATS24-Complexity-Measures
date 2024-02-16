import torch
import numpy as np
from learner.optimize_gd_learner import OptimizeGDLearner
from core.torch_dataset import TorchDataset

###############################################################################


class PredictLearner(OptimizeGDLearner):

    def __init__(self, model, batch_size, sampler=None):
        if(sampler is None):
            super().__init__(batch_size, shuffle=False)
        else:
            super().__init__(batch_size, sampler=sampler, shuffle=True)
        self._model = model

    def fit(self, data):

        self.__ref = None
        for key in data.keys():
            if(isinstance(data[key], torch.Tensor)):
                self.__ref = data[key]
                continue
            elif(isinstance(data[key], np.ndarray)):
                self.__ref = data[key]
                continue
        if(self.__ref is None):
            raise ValueError("ref must be either torch.tensor or np.ndarray")
        data = TorchDataset(data)
        return super().fit(data)

    def _begin(self):
        self._done = False
        self._y_pred = None

    def _meet_condition(self):
        if(not(self._done)):
            self._done = True
            return False
        return True

    def _optimize(self):

        for key in self._batch.keys():
            try:
                self._batch[key] = self._batch[key].to(self._model.device)
            except AttributeError:
                pass

        y_pred = self._model(self._batch)
        if(self._y_pred is None):
            self._y_pred = y_pred
        else:
            if(len(self._y_pred.shape)-1 == len(y_pred.shape)):
                y_pred = torch.unsqueeze(y_pred, dim=0)
            self._y_pred = torch.concat([self._y_pred, y_pred], dim=0)

    def _end(self):
        return self.torch_to_numpy(self.__ref, self._y_pred)
###############################################################################
