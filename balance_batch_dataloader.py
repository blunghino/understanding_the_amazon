import random

from torch.utils.data.dataloader import DataLoader, DataLoaderIter
from torch.utils.data.sampler import RandomSampler


class BalanceSampler(RandomSampler):

    def __init__(self, data_source, class_dict):
        self.num_samples = len(data_source)
        self.class_dict = class_dict

class BalanceDataLoaderIter(DataLoaderIter):

    def _next_indices(self):
        batch_size = min(self.samples_remaining, self.batch_size)
        # batch = [next(self.sample_iter) for _ in range(batch_size)]
        batch = []
        keys = list(self.sampler.class_dict.keys())
        for i in range(batch_size):
            randclass = random.choice(keys)
            batch.append(random.choice(self.sampler.class_dict[randclass]))

        self.samples_remaining -= len(batch)
        return batch

class BalanceDataLoader(DataLoader):

    def __iter__(self):
        return BalanceDataLoaderIter(self)