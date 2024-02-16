import torch
import numpy as np
from core.torch_dataset import TorchDataset


class TorchDatasetList(torch.utils.data.Dataset):

    def __init__(self, dataset_list, seed=0):
        self._dataset_list = list(dataset_list)
        for i in range(len(self._dataset_list)):
            if(not(isinstance(self._dataset_list[i], TorchDataset))):
                raise RuntimeError(
                    "The {}-th dataset is not a TorchDataset".format(i))

        self._permutation_list = []
        for i in range(1, len(self._dataset_list)):
            if(len(self._dataset_list[0]) > len(self._dataset_list[i])):
                perm = np.random.permutation(len(self._dataset_list[0]))
                perm = perm % len(self._dataset_list[i])
            else:
                perm = np.random.permutation(len(self._dataset_list[i]))
                perm = perm[:len(self._dataset_list[0])]
            self._permutation_list.append(perm)

    def set_mode(self, mode):
        for i in range(self._dataset_list):
            self._dataset_list.set_mode(mode)

    def get_mode(self):
        return self._dataset_list[0].get_mode()

    def __len__(self):
        # Getting the size of a dataset (of a given "mode")
        return len(self._dataset_list[0])

    def __getitem__(self, i):
        # Getting each example for a given mode
        item_dict = self._dataset_list[0].__getitem__(i)

        for j in range(1, len(self._dataset_list)):
            data = self._dataset_list[j]
            perm = self._permutation_list[j-1][i]
            item_dict.update(data.__getitem__(perm))

        return item_dict
