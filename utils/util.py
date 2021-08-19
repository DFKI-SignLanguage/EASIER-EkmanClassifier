import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, label_names, writer=None):
        self.writer = writer
        self.label_names = label_names

        keys_per_class = [k for k in keys if "_per_class" in k]
        self._data_per_class = pd.DataFrame(index=keys_per_class,
                                            columns=[lab + "_" + col_name for lab in self.label_names for col_name in
                                                     ['total', 'counts', 'average']])

        keys = [k for k in keys if "_per_class" not in k]
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])

        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
        for col in self._data_per_class.columns:
            self._data_per_class[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_per_class(self, key, data_dict, n=1):
        if self.writer is not None:
            self.writer.add_scalars(key, data_dict)
        for cls_key, v in data_dict.items():
            self._data_per_class[cls_key + "_total"][key] += v * n
            self._data_per_class[cls_key + "_counts"][key] += n
            self._data_per_class[cls_key + "_average"][key] = self._data_per_class[cls_key + "_total"][key] / \
                                                              self._data_per_class[cls_key + "_counts"][key]

    def result(self):
        avg = dict(self._data.average)
        col_avg_names = [lab + "_average" for lab in self.label_names]

        keys = self._data_per_class.index.values.tolist()
        vals = self._data_per_class[col_avg_names].values.tolist()
        d = {}
        for i in range(len(keys)):
            d[keys[i]] = vals[i]
        avg.update(d)

        return avg
