import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from inspect import ismethod


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 is_imbalanced_classes=False):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, is_imbalanced_classes)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split, is_imbalanced_classes):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        if is_imbalanced_classes:
            print("Weighted Random Sampler")
            methods = ["reorder_samples", "get_sample_weights"]
            for method in methods:
                assert (hasattr(self.dataset, method) and ismethod(getattr(self.dataset, method))), \
                    "Please implement {} method in your custom dataset class".format(method)

            self.dataset.reorder_samples(idx_full)

            idx_full_in_order = np.arange(self.n_samples)
            valid_idx = idx_full_in_order[0:len_valid]
            train_idx = np.delete(idx_full_in_order, np.arange(0, len_valid))

            train_sample_weights = self.dataset.get_sample_weights(train_idx)
            train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)

            valid_sample_weights = self.dataset.get_sample_weights(valid_idx)
            valid_sampler = WeightedRandomSampler(valid_sample_weights, len(valid_sample_weights), replacement=True)
        else:
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
