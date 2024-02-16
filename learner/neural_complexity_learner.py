import torch
import math
import logging
from module.module import Module
from learner.optimize_gd_learner import OptimizeGDLearner
from core.torch_dataset import TorchDataset
from learner.predict_learner import PredictLearner
from core.imbalanced_regression_sampler import ImbalancedRegressionSampler

###############################################################################


class NeuralComplexityLearner(OptimizeGDLearner):

    def __init__(self, model, lr, batch_size, epoch, model_writer, seed=0):

        self._model = model
        self._lr = lr
        self._batch_size = batch_size
        self._epoch_size = epoch
        self._model_writer = model_writer

        super().__init__(batch_size, sampler=None, shuffle=False, seed=seed)

        self._optim = torch.optim.Adam(
            self._model.parameters(), lr=self._lr)

        self._epoch = 0
        self.__sum_mean = None
        self.__loss_val = math.inf

        self._loss = Module("MAE")

        self._predict_learner = PredictLearner(self._model, self._batch_size)

    # ----------------------------------------------------------------------- #

    def fit(self, x_train_dict, y_train, x_val_dict, y_val):

        self._x_train_dict = x_train_dict
        self._y_train = y_train
        self._x_val_dict = x_val_dict
        self._y_val = y_val

        data_dict = {"y_train_batch": self._y_train, "step_train_": "train"}
        for param in self._x_train_dict.keys():
            data_dict[param+"_train_batch"] = self._x_train_dict[param]

        # We deal with imbalanced dataset
        self.sampler = ImbalancedRegressionSampler(
            self._y_train, bin_size=50, threshold=0.01)
        predict_sampler = ImbalancedRegressionSampler(
            self._y_val, bin_size=50, threshold=0.01)
        self._predict_learner.sampler = predict_sampler

        data = TorchDataset(data_dict)
        super().fit(data)

    # ----------------------------------------------------------------------- #

    def _meet_condition(self):
        self._epoch += 1
        if(self._epoch <= self._epoch_size):
            return False
        return True

    # ----------------------------------------------------------------------- #

    def _begin_epoch(self):

        self.__sum_loss = 0.0
        self.__sum_mean = None
        self.__sum_size = 0

    # ----------------------------------------------------------------------- #

    def _optimize(self):

        for key in self._batch.keys():
            try:
                self._batch[key] = self._batch[key].to(self._model.device)
            except AttributeError:
                pass
        y_train = self._batch["y"]
        yp_train = self._model(self._batch)

        loss = self._loss(yp_train, y_train)
        self._log["loss"] = loss

        self.__sum_loss += y_train.shape[0]*loss.item()
        self.__sum_size += y_train.shape[0]
        self.__sum_mean = self.__sum_loss/self.__sum_size
        self._log["mean loss"] = self.__sum_mean

        loss.backward()
        self._optim.step()
        self._optim.zero_grad()

    # ----------------------------------------------------------------------- #

    def _end_epoch(self):
        data_dict = {"step_train_": "predict"}
        for param in self._x_val_dict.keys():
            data_dict[param+"_train_batch"] = self._x_val_dict[param]
        yp_val = self._predict_learner.fit(data_dict)

        loss_val = self._loss(yp_val, self._y_val)
        if(loss_val < self.__loss_val):
            logging.info("{} < {}: Saving the model ...\n".format(
                loss_val, self.__loss_val))
            self.__loss_val = loss_val
            self.__save()
        else:
            logging.info("{} >= {}: Continuing ...\n".format(
                loss_val, self.__loss_val))

    # ----------------------------------------------------------------------- #

    def __save(self):
        state_dict = dict(self._model.state_dict())
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu().detach().numpy()
        self._model_writer.erase()
        self._model_writer.write(**state_dict)

###############################################################################
