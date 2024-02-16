import sys
import os
import h5py
import random
import torch
import warnings
import torchvision
import torchvision.datasets
import shutil
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import inspect


def call_fun(fun, kwargs):

    kwargs = dict(kwargs)

    fun_param = list(inspect.signature(fun).parameters.keys())
    for key in list(kwargs.keys()):
        if(key not in fun_param):
            del kwargs[key]

    return fun(**kwargs)


def get_label(label):
    label = label.numpy().astype(int)
    return label


def get_input(input):
    input = input.numpy().astype(np.float32)
    return input


def save(path, input_train, input_test, label_train, label_test):
    dataset_file = h5py.File(path, "w")
    dataset_file["x_train"] = input_train
    dataset_file["y_train"] = label_train
    dataset_file["x_test"] = input_test
    dataset_file["y_test"] = label_test


###############################################################################

def main():

    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    warnings.filterwarnings("ignore")

    arg_parser = ArgumentParser(
        description="generate a torchvision dataset")
    arg_parser.add_argument(
        "dataset", metavar="dataset", type=str,
        help="name of the dataset"
    )
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="path of the h5 dataset file"
    )
    arg_list = arg_parser.parse_args()
    dataset = arg_list.dataset
    path = arg_list.path

    if(os.path.exists(dataset)):
        input_label_train = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/train",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        input_label_test = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/test",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        test_size = len(input_label_test)
        train_size = len(input_label_train)
    else:
        dataset_fun = None
        _locals = locals()
        exec("dataset_fun = torchvision.datasets."+str(dataset),
             globals(), _locals)
        dataset_fun = _locals["dataset_fun"]
        kwargs = {
            "root": "./data-"+dataset,
            "train": True,
            "download": True,
            "split": "train",
            "transform": torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        }

        input_label_train = call_fun(dataset_fun, kwargs)
        if("train" in kwargs):
            kwargs["train"] = False
        if("split" in kwargs):
            kwargs["split"] = "test"
        input_label_test = call_fun(dataset_fun, kwargs)

        test_size = input_label_test.data.shape[0]
        train_size = input_label_train.data.shape[0]

        shutil.rmtree("./data-"+dataset)

    train_loader = DataLoader(
        input_label_train,
        batch_size=train_size)
    test_loader = DataLoader(
        input_label_test, batch_size=test_size)
    input_label_train = list(train_loader)
    input_label_test = list(test_loader)
    input_train = input_label_train[0][0]
    label_train = input_label_train[0][1]
    input_test = input_label_test[0][0]
    label_test = input_label_test[0][1]

    input_train = get_input(input_train)
    input_test = get_input(input_test)
    label_train = get_label(label_train)
    label_test = get_label(label_test)

    save(path, input_train, input_test, label_train, label_test)


if __name__ == "__main__":
    main()
