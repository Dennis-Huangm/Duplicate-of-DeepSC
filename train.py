# Denis
# coding:UTF-8
from contextlib import nullcontext
import sys

from mutual_info import *
from utils import *
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast


def train_p1(net, mi_model, X, valid_lens, opt, scaler):
    opt.zero_grad()
    autocast_context = autocast() if X.device.type == 'cuda' else nullcontext()
    with autocast_context:
        enc_output = PowerNormalize(net.transmitter(X, valid_lens))
        channel_output = net.channel.AWGN(enc_output, 0.1)
        joint, marg = sample_batch(enc_output, channel_output)
        loss_mi = -mutual_information(joint.detach(), marg.detach(), mi_model)

    scaler.scale(loss_mi).backward()
    # torch.nn.utils.clip_grad_norm_(mi_model.parameters(), 1)
    scaler.step(opt)
    scaler.update()
    # loss_mi.backward()
    # opt.step()
    return channel_output, enc_output


def train_p2(
    net, channel_output, enc_output, X, mi_model, dec_input, valid_lens, opt, scaler,
    sampling_prob=0.0  # Scheduled Sampling: 使用模型预测的概率
):
    loss = MaskedSoftmaxCELoss()
    target_valid_lens = torch.clamp(valid_lens - 1, min=1)
    opt.zero_grad()
    autocast_context = autocast() if channel_output.device.type == 'cuda' else nullcontext()
    with autocast_context:
        # Scheduled Sampling: 逐步从 Teacher Forcing 过渡到自回归解码
        if sampling_prob > 0 and torch.rand(1).item() < sampling_prob:
            # 使用自回归方式生成部分输入
            channel_dec = net.receiver.channel_decoder(channel_output)
            dec_state = net.receiver.transformer_decoder.init_state(channel_dec, valid_lens)
            mixed_input = dec_input.clone()
            for t in range(1, dec_input.size(1)):
                Y, dec_state = net.receiver.transformer_decoder(mixed_input[:, :t], dec_state)
                if torch.rand(1).item() < sampling_prob:
                    mixed_input[:, t] = Y[:, -1, :].argmax(dim=-1)
                # 重置 dec_state 以便下次使用完整序列
                dec_state = net.receiver.transformer_decoder.init_state(channel_dec, valid_lens)
            pred, _ = net.receiver(mixed_input, channel_output, valid_lens)
        else:
            pred, _ = net.receiver(dec_input, channel_output, valid_lens)
        
        joint, marg = sample_batch(enc_output, channel_output)
        mi_info = mutual_information(joint, marg, mi_model)
        l = loss(pred, X, target_valid_lens).mean() - 0.0009 * mi_info

    scaler.scale(l).backward()
    scaler.step(opt)
    scaler.update()
    return l.item(), mi_info.item()


def val_epoch(net, test_iter, device, vocab, snr):
    loss = MaskedSoftmaxCELoss()
    net.eval()
    metric = Accumulator(2)  # 统计损失训练总和
    pbar = tqdm(test_iter, desc="Testing", ascii=True, unit="batch")

    with torch.no_grad():
        for batch in pbar:
            src, valid_lens = [x.to(device) for x in batch]
            target_valid_lens = torch.clamp(valid_lens - 1, min=1)
            target, num_steps = src[:, 1:], src.shape[1] - 1
            noise_std = SNR_to_noise(snr)
            dec_X = torch.unsqueeze(
                torch.tensor(
                    [vocab["token_to_idx"]["<START>"]] * len(batch[1]),
                    dtype=src.dtype,
                    device=device,
                ),
                dim=1,
            )
            output, pred = [], []

            enc_output = PowerNormalize(net.transmitter(src, valid_lens))
            channel_enc = net.channel.AWGN(enc_output, noise_std)
            channel_dec = net.receiver.channel_decoder(channel_enc)
            dec_state = net.receiver.transformer_decoder.init_state(
                channel_dec, valid_lens
            )
            for _ in range(num_steps):
                Y, dec_state = net.receiver.transformer_decoder(dec_X, dec_state)
                dec_X = Y.argmax(dim=2)
                output.append(Y)
                pred.append(dec_X.type(torch.int32))

            output = torch.cat(output, dim=1)
            pred = torch.cat(pred, dim=1)
            loss_CE = loss(output, target, target_valid_lens).mean()
            metric.add(1, loss_CE)
        print("label：" + str(target[:10, :]))
        print("test预测结果：" + str(pred[:10, :]))  # 修复：移除 [1:] 对齐 target
    return metric[1] / metric[0]


def val_epoch1(net, test_iter, device):
    loss = MaskedSoftmaxCELoss()
    metric = Accumulator(2)  # 统计损失训练总和
    pbar = tqdm(test_iter, desc="Testing", ascii=True, unit="batch", file=sys.stdout)
    with torch.no_grad():
        for batch in pbar:
            src, valid_lens = [x.to(device) for x in batch]
            target_valid_lens = torch.clamp(valid_lens - 1, min=1)
            X, dec_input = src[:, 1:], src[:, :-1]  # 一个去除<bos>,一个去除<eos>
            pred = net(src, dec_input, valid_lens)
            l = loss(pred, X, target_valid_lens).mean()
            metric.add(1, l)
    return metric[1] / metric[0]
