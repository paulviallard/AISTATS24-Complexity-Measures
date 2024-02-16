import torch
import random

from module.module import Module

from learner.optimize_gd_learner import OptimizeGDLearner
from core.lambda_lr import LambdaLR
from core.choose_lr import ChooseLR
from core.sgld_optim import SGLD
from core.torch_dataset import TorchDataset
from core.torch_dataset_list import TorchDatasetList

###############################################################################


class ComplexityLearner():

    def __init__(
        self, model_init, model, comp, alpha, decay, batch_size, epoch,
        seed_list=[0, 0]
    ):
        self._model = model
        self._model_init = model_init
        self._comp = comp

        self._alpha = alpha
        self._decay = decay
        self._batch_size = batch_size
        self._epoch_size = epoch
        self.__seed_list = seed_list

        if(self._comp is None):
            self._comp_init = Module(
                "Complexity", "risk", 1.0, 0.5)
            self._epoch_init_size = random.randint(1, self._epoch_size)
        else:
            self._comp_init = None

        # ------------------------------------------------------------------- #

    def fit(self, x_optim, y_optim, x_test, y_test):

        # If the complexity does not contain the risk, we initialize the model
        # by minimizing the risk with SGLD for a random number of epochs
        if(self._comp_init is not None):
            learner_init = SingleComplexityLearner(
                self._model_init, self._model, self._comp_init,
                self._alpha, self._decay, self._batch_size,
                self._epoch_init_size, sgd=True, seed=self.__seed_list[0])
            learner_init.fit(x_optim, y_optim, x_test, y_test)
        else:
            # We run SGLD on the complexity measure
            learner_init = SingleComplexityLearner(
                self._model_init, self._model, self._comp,
                self._alpha, self._decay, self._batch_size,
                self._epoch_size, seed=self.__seed_list[1])
            learner_init.fit(x_optim, y_optim, x_test, y_test)


###############################################################################


class SingleComplexityLearner(OptimizeGDLearner):

    def __init__(
        self, model_init, model, comp, alpha, decay, batch_size, epoch,
        sgd=False, seed=0
    ):

        super().__init__(batch_size, shuffle=True, seed=seed)

        self._model = model
        self._model_init = model_init
        self._comp = comp

        self._alpha = alpha
        self._decay = decay
        self._epoch_size = epoch

        self._sgd = sgd

        if(not(self._sgd)):
            self._optim = SGLD(
                self._model.parameters(), lr=1.0, alpha=self._alpha)
        else:
            self._optim = torch.optim.SGD(
                self._model.parameters(), lr=1.0)
        if(comp.name not in [
            "dist_l2", "sum_fro", "dist_fro", "param_norm",
            "path_norm", "gap", "neural", "dist_neural",
            "dist_dist_l2", "dist_sum_fro", "dist_dist_fro",
            "dist_param_norm", "dist_path_norm", "dist_gap", "dist_l2"
        ]):
            self._lr_init = ChooseLR(
                self._optim, start_lr=1.0, factor=0.1, threshold=0.0001)
        elif(comp.name in ["gap", "dist_gap"]):
            self._lr_init = None
            self._epoch_size = 0
        else:
            self._lr_init = ChooseLR(
                self._optim, start_lr=1.0, factor=0.1,
                default_lr=0.0, threshold=0.01
            )

        self._scheduler = LambdaLR(self._optim, decay=self._decay)

        self.__sum_loss = None
        self.__sum_mean = None
        self.__sum_size = 0

        self._epoch = 0

    def fit(self, x_optim, y_optim, x_test, y_test):
        data_optim = TorchDataset(
            {"x_optim_train_batch": x_optim, "y_optim_train_batch": y_optim})
        data_test = TorchDataset(
            {"x_test_train_batch": x_test, "y_test_train_batch": y_test})
        data = TorchDatasetList([data_optim, data_test])

        super().fit(data)

    def _meet_condition(self):
        self._epoch += 1
        if(self._epoch <= self._epoch_size):
            return False
        return True

    def _begin_epoch(self):
        self.__lr_chosen = self._lr_init.step(self.__sum_mean)
        if(self.__lr_chosen == 1):
            self._epoch += 1
        if(self.__lr_chosen == -1 or self.__lr_chosen == 1):
            self._scheduler.step(self._epoch)

        self.__sum_loss = 0.0
        self.__sum_mean = None
        self.__sum_size = 0

    def _optimize(self):
        x_optim = self._batch["x_optim"]
        y_optim = self._batch["y_optim"]
        x_test = self._batch["x_test"]
        y_test = self._batch["y_test"]

        x_optim = x_optim.to(self._model.device)
        y_optim = y_optim.to(self._model.device)
        x_test = x_test.to(self._model.device)
        y_test = y_test.to(self._model.device)

        yp_optim = self._model({"x": x_optim})

        yp_test = self._model({"x": x_test})
        loss = self._comp(
            self._model_init, self._model, yp_optim, y_optim, yp_test, y_test)
        self._log["loss"] = loss

        self.__sum_loss += x_optim.shape[0]*loss.item()
        self.__sum_size += x_optim.shape[0]
        self.__sum_mean = self.__sum_loss/self.__sum_size
        self._log["mean loss"] = self.__sum_mean

        loss.backward()
        self._optim.step()
        self._optim.zero_grad()

    def _end_epoch(self):
        if(self.__lr_chosen == 0):
            self._epoch -= 1
