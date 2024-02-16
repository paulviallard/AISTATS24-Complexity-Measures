import argparse
import logging
import numpy as np
import os
import random
import torch

from learner.neural_complexity_learner import NeuralComplexityLearner
from module.model import Model
from core.writer import Writer

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
        "data_path", metavar="data_path", type=str,
        help="data_path")
    arg_parser.add_argument(
        "model_path", metavar="model_path", type=str,
        help="model_path")
    arg_parser.add_argument(
        "--seed", metavar="seed", default=0, type=int,
        help="seed")

    arg_parser.add_argument(
        "--batch_size", metavar="batch_size", default=1, type=int,
        help="batch_size")
    arg_parser.add_argument(
        "--val", metavar="val", default=0.1, type=float,
        help="val")
    arg_parser.add_argument(
        "--lr", metavar="lr", default=0.0, type=float,
        help="lr")
    arg_parser.add_argument(
        "--epoch", metavar="epoch", default=1, type=int,
        help="epoch")

    # ----------------------------------------------------------------------- #

    arg_list = arg_parser.parse_known_args()[0]

    data = arg_list.data
    data_path = arg_list.data_path
    model_path = arg_list.model_path
    seed = arg_list.seed*3

    batch_size = arg_list.batch_size
    val = arg_list.val
    lr = arg_list.lr
    epoch = arg_list.epoch

    # ----------------------------------------------------------------------- #

    device_list = ["cuda", "cpu"]

    # ----------------------------------------------------------------------- #

    writer = Writer(
        os.path.join(os.path.dirname(__file__), data_path), mode="h5")
    writer.load()
    data = dict(writer.file_dict)
    del writer

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    permutation = np.arange(data[list(data.keys())[0]].shape[0])
    np.random.shuffle(permutation)

    for key in data:
        data[key] = np.array(data[key])
        data[key] = data[key][permutation]

    m = data[list(data.keys())[0]].shape[0]
    m = m-int(val*m)
    x_train_dict = {}
    x_val_dict = {}
    i = 0

    while("param_"+str(i) in data):
        x_train_dict["param_"+str(i)] = data["param_"+str(i)][:m]
        x_val_dict["param_"+str(i)] = data["param_"+str(i)][m:]
        i += 1
    y = np.expand_dims(np.abs(data["risk_val"]-data["risk_train"]), axis=1)
    y_train = y[:m]
    y_val = y[m:]

    # ----------------------------------------------------------------------- #

    model_writer = Writer(
        os.path.join(os.path.dirname(__file__), model_path), mode="h5")

    # ----------------------------------------------------------------------- #

    logging.info("We learn the neural complexity measure...\n")

    model = Model("MNISTComplexityMeasure", seed=seed+1)
    model.to(device_list)

    learner = NeuralComplexityLearner(
        model, lr, batch_size, epoch, model_writer, seed=seed+2)
    learner.fit(x_train_dict, y_train, x_val_dict, y_val)

    del model_writer

###############################################################################


if __name__ == "__main__":
    main()
