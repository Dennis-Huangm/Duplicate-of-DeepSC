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
    opt_mi = torch.optim.Adam(mi_model.parameters(),lr)
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


if __name__ == '__main__':
    with open('./content/vocab.json', 'rb') as file:
        vocab = json.load(file)

    batch_size, vocab_size = 128, len(vocab['token_to_idx'])
    train_datasets = EurDataset()
    test_datasets = EurDataset(split='test')
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=batch_size, collate_fn=collate_data)
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=batch_size, collate_fn=collate_data)

    num_hiddens, num_layers, dropout = 128, 3, 0.1
    learning_rate, epochs = 1e-3, 50
    key_size, query_size, value_size = 128, 128, 128
    ffn_num_input, ffn_num_hiddens, num_heads = 128, 256, 8
    norm_shape = [num_hiddens]
    transceiver = Transceiver(num_layers, vocab_size, key_size, query_size,
                              value_size, num_hiddens, norm_shape, ffn_num_input,
                              ffn_num_hiddens, num_heads, dropout)
    mi_net = Mine()

    run(transceiver, mi_net, train_loader, test_loader, learning_rate, epochs, try_gpu())
