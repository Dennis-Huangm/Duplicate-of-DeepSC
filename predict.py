# Denis
# coding:UTF-8
from mutual_info import *
from utils import *
from datasets import EurDataset, collate_data
import json
from models import Transceiver
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from train import train_p1, train_p2, val_epoch
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import sys
import argparse
import torch


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='the epochs of training')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--ffn-num-input', type=int, default=128, help='ffn\'s input dim')
    parser.add_argument('--ffn-num-hiddens', type=int, default=256, help='the hidden size of transformers\'s ffn')
    parser.add_argument('--num-hiddens', type=int, default=128, help='the dimension of channel encoding')
    parser.add_argument('--key-size', type=int, default=128, help='the dimension of key')
    parser.add_argument('--query-size', type=int, default=128, help='the dimension of query')
    parser.add_argument('--value-size', type=int, default=128, help='the dimension of value')
    parser.add_argument('--num-layers', type=int, default=3, help='the layers of encoder and decoder')
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--lr', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--num_heads', type=int, default=8, help='multiple head of attention')
    parser.add_argument('--norm-shape', type=list, default=[128])
    parser.add_argument('--vocab', type=str, default='./content/vocab.json')
    parser.add_argument('--save-csv', type=bool, default=False, help='save the result as csv file')
    parser.add_argument('--save-img', type=bool, default=False, help='save the loss arc as img')
    return parser.parse_args()


def main(opt):
    vocab, ffn_num_input, ffn_num_hiddens, key_size, query_size, value_size, num_layers, dropout, lr, num_heads, \
    norm_shape, save_csv, save_img, num_hiddens = opt.vocab, opt.ffn_num_input, opt.ffn_num_hiddens, opt.key_size, \
                                                  opt.query_size, opt.value_size, opt.num_layers, opt.dropout, \
                                                  opt.lr, opt.num_heads, opt.norm_shape, opt.save_csv, \
                                                  opt.save_img, opt.num_hiddens

    test_datasets = EurDataset(split='test')
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_data)
    with open(opt.vocab, 'rb') as file:
        vocab = json.load(file)
        vocab_size = len(vocab['token_to_idx'])

    transceiver = Transceiver(num_layers, vocab_size, key_size, query_size,
                              value_size, num_hiddens, norm_shape, ffn_num_input,
                              ffn_num_hiddens, num_heads, dropout).to(opt.device)
    transceiver.load_state_dict(torch.load('model.pt'))
    loss = MaskedSoftmaxCELoss()
    l=val_epoch(transceiver, test_loader, opt.device, loss, vocab)
    print(l)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
