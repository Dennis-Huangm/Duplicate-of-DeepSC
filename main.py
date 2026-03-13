# Denis
# coding:UTF-8
import argparse
import json

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import EurDataset, collate_data
from models import Transceiver
from mutual_info import *
from train import train_p1, train_p2, val_epoch
from utils import *


def run(net, mi_model, train_iter, test_iter, lr, num_epochs, device, vocab, checkpoint_path, model_config):
    def xavier_init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    scaler_enabled = device.type == 'cuda'
    scaler1, scaler2 = GradScaler(enabled=scaler_enabled), GradScaler(enabled=scaler_enabled)
    writer = SummaryWriter(log_dir=resolve_repo_path('./runs'))
    metric = Accumulator(3)
    opt_global = torch.optim.AdamW(net.parameters(), lr, eps=1e-7)
    opt_mi = torch.optim.Adam(mi_model.parameters(), lr)

    net.apply(xavier_init_weights)
    net.to(device)
    mi_model.to(device)

    for epoch in range(num_epochs):
        net.train()
        mi_model.train()
        sampling_prob = min(0.5, max(0, (epoch - 10) / 40))
        pbar = tqdm(train_iter, ascii=True, unit="batch")
        for batch in pbar:
            src, valid_lens = [x.to(device) for x in batch]
            target, dec_input = src[:, 1:], src[:, :-1]
            channel_output, enc_output = train_p1(net, mi_model, src, valid_lens, opt_mi, scaler1)
            loss, mi_info = train_p2(
                net,
                channel_output,
                enc_output,
                target,
                mi_model,
                dec_input,
                valid_lens,
                opt_global,
                scaler2,
                sampling_prob,
            )
            with torch.no_grad():
                metric.add(1, mi_info, loss)
            pbar.set_description(
                'Training:epoch {0}/{1} loss:{2:.3f} mi_info:{3:.3f} sp:{4:.2f}'.format(
                    epoch + 1,
                    num_epochs,
                    loss,
                    mi_info,
                    sampling_prob,
                )
            )
        val_loss = val_epoch(net, test_iter, device, vocab, 12)
        print(
            "=============== Train_Loss:{0:.3f} mi_info:{1:.3f} Test_loss:{2:.3f} ===============\n".format(
                metric[2] / metric[0],
                metric[1] / metric[0],
                val_loss,
            )
        )
        writer.add_scalar('loss', metric[2] / metric[0], epoch + 1)
        writer.add_scalar('mutual_info', metric[1] / metric[0], epoch + 1)
        metric.reset()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'model_config': model_config,
        'vocab_size': len(vocab['token_to_idx']),
        'start_token_id': vocab['token_to_idx']['<START>'],
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60, help='the epochs of training')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device for training and evaluation')
    parser.add_argument('--ffn-num-input', type=int, default=128, help='ffn\'s input dim')
    parser.add_argument('--ffn-num-hiddens', type=int, default=256, help='the hidden size of transformers\'s ffn')
    parser.add_argument('--num-hiddens', type=int, default=128, help='the dimension of channel encoding')
    parser.add_argument('--key-size', type=int, default=128, help='the dimension of key')
    parser.add_argument('--query-size', type=int, default=128, help='the dimension of query')
    parser.add_argument('--value-size', type=int, default=128, help='the dimension of value')
    parser.add_argument('--num-layers', type=int, default=3, help='the layers of encoder and decoder')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_heads', type=int, default=8, help='multiple head of attention')
    parser.add_argument('--norm-shape', nargs='+', type=int, default=[128], help='layer norm shape values')
    parser.add_argument('--vocab', type=str, default='./content/vocab.json', help='path to vocab json')
    parser.add_argument('--checkpoint-path', type=str, default='./artifacts/model.pt', help='path to save the trained checkpoint')
    parser.add_argument('--save-csv', action='store_true', help='save the result as csv file')
    parser.add_argument('--save-img', action='store_true', help='save the loss arc as img')
    return parser.parse_args()



def main(opt):
    device = resolve_device(opt.device)
    data_dir = resolve_repo_path('./content')
    vocab_path = resolve_repo_path(opt.vocab)
    checkpoint_path = resolve_repo_path(opt.checkpoint_path)

    train_datasets = EurDataset(data_dir=data_dir)
    test_datasets = EurDataset(split='test', data_dir=data_dir)
    train_loader = DataLoader(train_datasets, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_data)
    test_loader = DataLoader(test_datasets, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_data)
    with vocab_path.open('r', encoding='utf-8') as file:
        vocab = json.load(file)
        token_to_idx = validate_vocab(vocab)
        vocab_size = len(token_to_idx)

    model_config = build_transceiver_config(opt, vocab_size)
    transceiver = Transceiver(**model_config)
    mi_net = Mine()

    run(transceiver, mi_net, train_loader, test_loader, opt.lr, opt.epochs, device, vocab, checkpoint_path, model_config)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
