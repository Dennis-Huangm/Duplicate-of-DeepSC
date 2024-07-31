# Denis
# coding:UTF-8
from mutual_info import *
from utils import *
from tqdm import tqdm
import sys
from torch.cuda.amp import autocast


def train_p1(net, mi_model, X, valid_lens, opt, scaler):
    opt.zero_grad()
    with autocast(enabled=False):
        enc_output = net.transmitter(X, valid_lens)
        channel_output = PowerNormalize(net.channel.add_awgn(enc_output, 12))
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
    with autocast(enabled=False):
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


def val_epoch(net, test_iter, device, mi_model, loss):
    metric = Accumulator(2)  # 统计损失训练总和
    pbar = tqdm(test_iter, desc='Testing', ascii=True, unit="batch", file=sys.stdout)
    with torch.no_grad():
        for batch in pbar:
            src, valid_lens = [x.to(device) for x in batch]
            X, dec_input = src[:, 1:], src[:, :-1]  # 一个去除<bos>,一个去除<eos>
            with autocast(enabled=False):
                pred, channel_output, enc_output = net(X, dec_input, valid_lens)
                joint, marg = sample_batch(enc_output, channel_output)
                loss_mi = mutual_information(joint, marg, mi_model)
                l = loss(pred, X, valid_lens).mean() - 0.0009 * loss_mi
            metric.add(1, l)
    return metric[1] / metric[0]
