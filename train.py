# Denis
# coding:UTF-8
from mutual_info import *
from utils import *
from tqdm import tqdm
import sys
from torch.cuda.amp import autocast
import torch


def train_p1(net, mi_model, X, valid_lens, opt, scaler):
    opt.zero_grad()
    with autocast():
        enc_output = net.transmitter(X, valid_lens)
        channel_output = PowerNormalize(net.channel.add_AWGN(enc_output, 12))
        # print(check_snr(enc_output, net.channel.add_AWGN(enc_output, 12)))
        joint, marg = sample_batch(enc_output, channel_output)
        loss_mi = -mutual_information(joint, marg, mi_model)

    scaler.scale(loss_mi).backward(retain_graph=True)
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(mi_model.parameters(), 1)
    scaler.step(opt)
    scaler.update()
    return channel_output, enc_output


def train_p2(net, channel_output, enc_output, X, mi_model, dec_input, valid_lens, opt, loss, scaler):
    opt.zero_grad()
    with autocast():
        pred, _ = net.receiver(dec_input, channel_output, valid_lens)
        joint, marg = sample_batch(enc_output, channel_output)
        mi_info = mutual_information(joint, marg, mi_model)
        l = loss(pred, X, valid_lens).mean() - 0.0009 * mi_info

    scaler.scale(l).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    scaler.step(opt)
    scaler.update()
    return l.item(), mi_info.item()


def val_epoch(net, test_iter, device, loss, vocab):
    net.eval()
    metric = Accumulator(2)  # 统计损失训练总和
    pbar = tqdm(test_iter, desc='Testing', ascii=True, unit="batch", file=sys.stdout)

    with torch.no_grad():
        for batch in pbar:
            src, valid_lens = [x.to(device) for x in batch]
            X, num_steps = src[:, 1:], src.shape[1] - 1
            dec_X = torch.unsqueeze(torch.tensor(
                [vocab["token_to_idx"]['<START>']] * len(batch[1]), device=device), dim=1)
            output, pred = [], []
            with autocast():
                enc_output = net.transmitter(X, valid_lens)
                channel_enc = PowerNormalize(net.channel.add_AWGN(enc_output, 12))
                channel_dec = net.receiver.channel_decoder(channel_enc)
                dec_state = net.receiver.transformer_decoder.init_state(channel_dec, valid_lens)
                for _ in range(num_steps):
                    Y, dec_state = net.receiver.transformer_decoder(dec_X, dec_state)
                    dec_X = Y.argmax(dim=2)
                    output.append(Y)
                    pred.append(dec_X.type(torch.int32))

            output = torch.cat(output, dim=1)
            pred = torch.cat(pred, dim=1)
            loss_CE = loss(output, X, valid_lens).mean()
            metric.add(1, loss_CE)
    return metric[1] / metric[0]
