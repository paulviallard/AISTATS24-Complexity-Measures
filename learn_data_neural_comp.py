import argparse
from h5py import File
import logging
import numpy as np
import os
import random
import torch

from module.module import Module
from learner.data_neural_complexity_learner import DataNeuralComplexityLearner
from module.model import Model

###############################################################################


def main():
    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="path csv")
    arg_parser.add_argument(
        "--seed", metavar="seed", default=0, type=int,
        help="seed")

    # SGD
    arg_parser.add_argument(
        "--batch_size", metavar="batch_size", default=1, type=int,
        help="batch_size")
    arg_parser.add_argument(
        "--val", metavar="val", default=0.1, type=float,
        help="val")
    arg_parser.add_argument(
        "--decay", metavar="decay", default=0.0, type=float,
        help="decay")
    arg_parser.add_argument(
        "--epoch", metavar="epoch", default=1, type=int,
        help="epoch")
    arg_parser.add_argument(
        "--save_size", metavar="save_size", default=10, type=int,
        help="save_size")

    # ----------------------------------------------------------------------- #

    arg_list = arg_parser.parse_known_args()[0]

    data = arg_list.data
    path = arg_list.path
    seed = arg_list.seed*3

    # SGLD
    batch_size = arg_list.batch_size
    val = arg_list.val
    decay = arg_list.decay
    epoch = arg_list.epoch
    save_size = arg_list.save_size

    # ----------------------------------------------------------------------- #

    device_list = ["cuda", "cpu"]

    # ----------------------------------------------------------------------- #

    data = File(os.path.join("data", data+".h5"), "r")
    path = os.path.join(os.path.dirname(__file__), path)

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    x_mean, x_std = Module("MeanStd")(x_train)
    x_train = Module("StandardScaler")(x_train, x_mean=x_mean, x_std=x_std)
    y_train = np.expand_dims(y_train, 1)
    y_train = y_train.astype(np.int64)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    permutation = np.arange(x_train.shape[0])
    np.random.shuffle(permutation)
    x_train = x_train[permutation]
    y_train = y_train[permutation]

    m = len(x_train)-int(val*len(x_train))

    # We take the two sets
    x_val = x_train[m:]
    y_val = y_train[m:]
    x_train = x_train[:m]
    y_train = y_train[:m]

    # ----------------------------------------------------------------------- #
    # We learn the models
    logging.info("We learn the models...\n")

    model = Model("MNISTModel", seed=seed+1)
    model.to(device_list)

    learner = DataNeuralComplexityLearner(
        path, model, decay, batch_size, epoch, save_size,
        seed=seed+2)
    learner.fit(x_train, y_train, x_val, y_val)

###############################################################################


if __name__ == "__main__":
    main()
