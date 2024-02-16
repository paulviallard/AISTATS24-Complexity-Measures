from re import sub
import torch
import copy


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_dict):

        self._dataset_key = list(dataset_dict.keys())

        # Removing the mode in self._dataset_key
        new_dataset_key = set()
        for key in self._dataset_key:
            new_key = sub("_[^_]*_[^_]*$", "", key)
            new_dataset_key.add(new_key)
        self._dataset_key = list(new_dataset_key)

        self._dataset_dict = copy.deepcopy(dataset_dict)
        self._mode = "train"

    def keys(self, mode=None, all=True):
        if(mode is None):
            mode = self._mode

        key_list = []
        for key in self._dataset_key:

            key_mode = key+"_"+self._mode
            key_all = key+"__"
            key_mode_all = key+"_"+self._mode+"_"
            key_mode_batch = key+"_"+self._mode+"_batch"
            key_all_batch = key+"__batch"

            if(key_mode in self._dataset_dict.keys()):
                key_list.append(key_mode)
            if(all and key_all in self._dataset_dict.keys()):
                key_list.append(key_all)
            if(all and key_mode_all in self._dataset_dict.keys()):
                key_list.append(key_mode_all)
            if(key_mode_batch in self._dataset_dict.keys()):
                key_list.append(key_mode_batch)
            if(all and key_all_batch in self._dataset_dict.keys()):
                key_list.append(key_all_batch)

        return key_list

    def set_mode(self, mode):
        # Setting the mode of the dataset
        self._mode = mode

    def get_mode(self):
        # Getting the mode of the dataset
        return self._mode

    def __len__(self):
        # Getting the size of a dataset (of a given "mode")
        return len(self._dataset_dict[self.keys(all=False)[0]])

    def __getitem__(self, i):
        # Getting each example for a given mode
        item_dict = {}

        for key in self.keys():

            new_key_batch = sub("_batch$", "", key)
            new_key = sub("_[^_]*_[^_]*$", "", key)

            # If there is the "batch" flag
            if(key != new_key_batch):
                # We get the ith item
                item = self._dataset_dict[key][i]
            else:
                # Otherwise, we get everything
                item = self._dataset_dict[key]

            # If we have an example, we transform the example before
            if(isinstance(item, torch.Tensor)):
                item_dict[new_key] = item.clone().detach()
            elif(type(item).__module__ == "numpy"):
                item_dict[new_key] = torch.tensor(item)
            else:
                item_dict[new_key] = item

        return item_dict
