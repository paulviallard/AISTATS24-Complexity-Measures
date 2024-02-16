import torch
import logging

from module.module import Module
from core.lambda_lr import LambdaLR
from core.choose_lr import ChooseLR
from learner.optimize_gd_learner import OptimizeGDLearner
from core.torch_dataset import TorchDataset
from learner.predict_learner import PredictLearner
from core.writer import Writer

###############################################################################


class DataNeuralComplexityLearner(OptimizeGDLearner):

    def __init__(
        self, path, model, decay, batch_size, epoch, save_size, seed=0
    ):

        self._writer_path = path
        self._model = model
        self._decay = decay
        self._batch_size = batch_size
        self._epoch_size = epoch
        self._save_size = save_size

        super().__init__(batch_size, shuffle=True, seed=seed)

        self._optim = torch.optim.SGD(
            self._model.parameters(), lr=1.0)
        self._lr_init = ChooseLR(
            self._optim, start_lr=1.0, end_lr=1e-10, default_lr=0.1,
            factor=0.1, threshold=0.0001)
        self._scheduler = LambdaLR(self._optim, decay=self._decay)

        self._epoch = 0
        self.__sum_mean = None
        self.__lr_chosen = 0

        self._loss = Module("BoundedCrossEntropy")
        self._softmax = Module("Softmax")
        self._risk = Module("ZeroOne")
        self._writer = Writer(self._writer_path, mode="h5")

        self._predict_learner = PredictLearner(self._model, self._batch_size)

    # ----------------------------------------------------------------------- #

    def fit(self, x_train, y_train, x_val, y_val):

        self._x_train = x_train
        self._y_train = y_train
        self._x_val = x_val
        self._y_val = y_val

        data = TorchDataset(
            {"x_train_batch": x_train, "y_train_batch": y_train})
        super().fit(data)

    # ----------------------------------------------------------------------- #

    def _begin(self):
        self._save()

    # ----------------------------------------------------------------------- #

    def _meet_condition(self):
        self._epoch += 1
        if(self._epoch <= self._epoch_size):
            return False

        self._save()
        return True

    # ----------------------------------------------------------------------- #

    def _begin_epoch(self):

        self.__lr_chosen = self._lr_init.step(self.__sum_mean)

        if(self.__lr_chosen != 0 and
           (self._epoch-1) % int(self._epoch_size/self._save_size) == 0):
            self._save()

        if(self.__lr_chosen == 1):
            self._epoch += 1
        if(self.__lr_chosen == -1 or self.__lr_chosen == 1):
            self._scheduler.step(self._epoch)

        self.__sum_loss = 0.0
        self.__sum_mean = None
        self.__sum_size = 0

    # ----------------------------------------------------------------------- #

    def _save(self):
        logging.info("Saving the model ...\n")

        yp_train = self._predict_learner.fit({"x_train_batch": self._x_train})
        yp_train = self._softmax(yp_train, dim=1)
        yp_val = self._predict_learner.fit({"x_train_batch": self._x_val})
        yp_val = self._softmax(yp_val, dim=1)

        risk_train = self._risk(yp_train, self._y_train)
        risk_val = self._risk(yp_val, self._y_val)
        loss_train = self._loss(yp_train, self._y_train)
        loss_val = self._loss(yp_val, self._y_val)

        norm = 0.0
        for param in self._model.parameters():
            norm += torch.norm(param)**2.0
        norm = torch.sqrt(norm).cpu().detach().numpy()

        writer_dict = {}
        for i, param in enumerate(self._model.parameters()):
            writer_dict["param_"+str(i)] = param.cpu().detach().numpy()/norm
        writer_dict["risk_train"] = risk_train
        writer_dict["risk_val"] = risk_val
        writer_dict["loss_train"] = loss_train
        writer_dict["loss_val"] = loss_val
        writer_dict["norm"] = norm
        self._writer.write(**writer_dict)

        logging.info("Risk gap={:.4f}, Loss gap={:.4f}\n".format(
            risk_val-risk_train, loss_val-loss_train))
        logging.info("Train risk={:.4f}, Test risk={:.4f}\n".format(
            risk_train, risk_val))
        logging.info("Train loss={:.4f}, Test loss={:.4f}\n".format(
            loss_train, loss_val))

    # ----------------------------------------------------------------------- #

    def _optimize(self):

        for key in self._batch.keys():
            try:
                self._batch[key] = self._batch[key].to(self._model.device)
            except AttributeError:
                pass

        y_train = self._batch["y"]
        yp_train = self._model(self._batch)
        yp_train = torch.softmax(yp_train, dim=1)

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
        if(self.__lr_chosen == 0):
            self._epoch -= 1

###############################################################################
