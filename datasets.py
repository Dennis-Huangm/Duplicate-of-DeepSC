# coding:UTF-8
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EurDataset(Dataset):
    def __init__(self, split='train'):
        with open('./content/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_data(batch):
    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))  # get the max length of sentence in current batch
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)
    valid_lens = torch.zeros(len(batch))

    for i, sent in enumerate(sort_by_len):
        length = valid_lens[i] = len(sent) - 1
        sents[i, :length + 1] = sent  # padding the questions

    return torch.from_numpy(sents), valid_lens


if __name__ == '__main__':
    batch_size = 128
    train_datasets = EurDataset()
    test_datasets = EurDataset(split='test')
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size, collate_fn=collate_data)
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=batch_size, collate_fn=collate_data)
