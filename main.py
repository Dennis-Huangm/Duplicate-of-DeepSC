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


def run(net, mi_model, train_iter, test_iter, lr, num_epochs, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    scaler1, scaler2 = GradScaler(), GradScaler()
    writer = SummaryWriter('./logs')
    metric = Accumulator(3)  # 统计损失训练总和
    net.apply(xavier_init_weights)
    net.to(device)
    mi_model.to(device)
    net.train()
    mi_model.train()

    opt_global = torch.optim.AdamW(net.parameters(), lr, eps=1e-7)
    opt_mi = torch.optim.Adam(mi_model.parameters(), lr)
    CE_loss = MaskedSoftmaxCELoss()
    for epoch in range(num_epochs):
        pbar = tqdm(train_iter, ascii=True, unit="batch", file=sys.stdout)
        for batch in pbar:
            src, valid_lens = [x.to(device) for x in batch]
            X, dec_input = src[:, 1:], src[:, :-1]  # 一个去除<bos>,一个去除<eos>
            channel_output, enc_output = train_p1(net, mi_model, X, valid_lens, opt_mi, scaler1)
            total_loss, mi_info = train_p2(net, channel_output, enc_output, X, mi_model, dec_input,
                                           valid_lens, opt_global, CE_loss, scaler2)
            with torch.no_grad():
                metric.add(1, mi_info, total_loss)
            pbar.set_description(
                'Training:epoch {0}/{1} loss:{2:.3f} mi_info:{3:.3f}'.format(epoch + 1, num_epochs, total_loss,
                                                                             mi_info))
        val_loss = val_epoch(net, test_iter, device, mi_model, CE_loss)
        print("=============== Train_Loss:{0:.3f} mi_info:{1:.3f} Test_loss:{2:.3f} ===============\n".format(
            metric[2] / metric[0], metric[1] / metric[0], val_loss))
        writer.add_scalar('loss', metric[2] / metric[0], epoch + 1)
        writer.add_scalar('mutual_info', metric[1] / metric[0], epoch + 1)
        metric.reset()


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
    parser.add_argument('--save-csv', action='store_true', help='save the result as csv file')
    parser.add_argument('--save-img', action='store_true', help='save the loss arc as img')
    return parser.parse_args()


def main(opt):
    opt = parse_opt()
    vocab, ffn_num_input, ffn_num_hiddens, key_size, query_size, value_size, num_layers, dropout, lr, num_heads, \
    norm_shape, save_csv, save_img, num_hiddens = opt.vocab, opt.ffn_num_input, opt.ffn_num_hiddens, opt.key_size, \
                                                  opt.query_size, opt.value_size, opt.num_layers, opt.dropout, \
                                                  opt.lr, opt.num_heads, opt.norm_shape, opt.save_csv, \
                                                  opt.save_img, opt.num_hiddens

    train_datasets = EurDataset()
    test_datasets = EurDataset(split='test')
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_data)
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_data)
    with open(opt.vocab, 'rb') as file:
        vocab = json.load(file)
        vocab_size = len(vocab['token_to_idx'])

    transceiver = Transceiver(num_layers, vocab_size, key_size, query_size,
                              value_size, num_hiddens, norm_shape, ffn_num_input,
                              ffn_num_hiddens, num_heads, dropout)
    mi_net = Mine()
    run(transceiver, mi_net, train_loader, test_loader, opt.lr, opt.epochs, opt.device)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
