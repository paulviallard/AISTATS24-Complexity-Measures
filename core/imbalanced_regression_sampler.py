import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler


class ImbalancedRegressionSampler(WeightedRandomSampler):

    def __init__(self, y, bin_size=None, threshold=1.0):

        if(isinstance(y, np.ndarray)):
            y = torch.tensor(y)
        if(bin_size is None):
            bin_size = len(y)

        bin_list = torch.linspace(
            torch.min(y), torch.max(y), bin_size)
        y_bin = torch.bucketize(y, bin_list)
        counts = torch.bincount(y_bin[:, 0])

        stabilized = False
        while(not(stabilized)):

            i = 0
            stabilized = True
            while(i < len(torch.unique(y_bin))):

                counts = torch.bincount(y_bin[:, 0])
                c = counts/torch.sum(counts)
                if(c[i] > 0.0 and c[i] <= threshold and
                   i == len(torch.unique(y_bin))-1):
                    y_bin[y_bin == i] = y_bin[y_bin == i]-1
                    stabilized = False
                elif(c[i] > 0.0 and c[i] <= threshold):
                    y_bin[y_bin > i] = y_bin[y_bin > i]-1
                    stabilized = False
                else:
                    i += 1

            counts = torch.bincount(y_bin[:, 0])
            c = counts/torch.sum(counts)

        y_weights = np.zeros(y_bin.shape)
        for i in np.unique(y_bin):
            y_weights[y_bin == i] = 1.0/c[i]

        super().__init__(y_weights[:, 0], len(y))
