# coding:UTF-8
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils import resolve_repo_path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EurDataset(Dataset):
    def __init__(self, split='train', data_dir=None):
        valid_splits = {'train', 'test', 'testC'}
        if split not in valid_splits:
            raise ValueError(f'Unsupported split: {split}')
        self.data_dir = resolve_repo_path(data_dir or './content')
        self.data_path = self.data_dir / f'{split}_data.pkl'
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_data(batch):
    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)
    valid_lens = torch.zeros(len(batch), dtype=torch.int64)

    for i, sent in enumerate(sort_by_len):
        length = valid_lens[i] = len(sent)
        sents[i, :length] = sent

    return torch.from_numpy(sents), valid_lens


if __name__ == '__main__':
    batch_num = 1
    train_datasets = EurDataset()
    test_datasets = EurDataset(split='test')
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=batch_num, collate_fn=collate_data)
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=batch_num, collate_fn=collate_data)
    print(len(train_datasets))
    for u, v in train_loader:
        print(u.shape)
        print(u)
        print(v)
        break
