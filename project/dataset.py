import numpy as np
import os
import torch
from util import log
import torchvision
import pickle


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, ids, path, name='default', max_examples=None, is_train=True,
                 filename='data.hy'):
        self._ids = ids
        self.name = name
        self.is_train = is_train

        if max_examples is not None:
            self._ids = self._ids[:max_examples]

        log.info("Reading %s ... " % (os.path.join(path, filename)))

        try:
            with open(os.path.join(path, filename), 'rb') as f:
                self.data = pickle.load(f)
        except IOError:
            raise IOError('Dataset not found!')
        log.info('Reading Done: %s' % (os.path.join(path, filename)))

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, item):
        # pre-processing and data augmentation
        id_ = self._ids[item]
        img = self.data[id_]['image']/255.
        q = self.data[id_]['question'].astype(np.float32)
        a = self.data[id_]['answer'].astype(np.float32)
        return img, q, a

    @property
    def ids(self):
        return self._ids

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def get_data_info():
    return np.array([128, 128, 3, 11, 10])


def get_conv_info():
    return np.array([24, 24, 24, 24])


def create_default_splits(path):
    ids = all_ids(path)
    n = len(ids)

    num_train = int(n * 0.7)  # 70%  for train
    num_valid = int(n * 0.15)  # 15 % for validation

    dataset_train = Dataset(ids[:num_train], path, name='train', is_train=True)
    dataset_valid = Dataset(ids[num_train:num_train+num_valid], path, name='valid', is_train=False)
    dataset_test = Dataset(ids[num_train+num_valid:], path, name='test', is_train=False)

    return dataset_train, dataset_valid, dataset_test


def all_ids(path):
    id_filename = 'id.txt'
    id_txt = os.path.join(path, id_filename)
    try:
        with open(id_txt, 'r') as fp:
            _ids = [s.strip() for s in fp.readlines() if s]
    except IOError:
        raise IOError('Dataset not found!')
    np.random.shuffle(_ids)
    return _ids
