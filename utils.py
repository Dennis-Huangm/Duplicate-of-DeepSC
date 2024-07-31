# Denis
# coding:UTF-8
import torch
from torch import nn
from torch.nn import functional


def sequence_mask(X, valid_len, value=0.0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    def forward(self, pred, label, valid_len):  # 函数的重载
        # valid_len的形状为：(batch_size,)
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)  # 设置掩码，由1和0组成
        self.reduction = 'none'  # 不进行任何减少操作，返回与输入形状相同的张量
        unweighted_loss = super().forward(pred.permute(0, 2, 1).float(), label)  # 将预测维度放在中间，在loss后消除vocab_size的维度
        # 相当于是消除了(batch_size,num_classes)中的num_classes
        weighted_loss = (unweighted_loss * weights).mean(1)  # 对一整个句子即一个batch的各个时间步的loss取平均，消除该维度
        return weighted_loss


def masked_softmax(X, valid_lens):
    # 输入X为(batch_size,num_query,num_kvpair(num_steps))
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    if valid_lens is None:
        return functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:  # 等于1的情况一般是编码器encoder的遮蔽，用来屏蔽padding
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            """将每个batch的有效长度复制num_query份，在自注意力中即为num_step个（为一列），之后在函数中经广播扩展每一行"""
        else:
            valid_lens = valid_lens.reshape(-1)  # 等于2的时候一般是transformer的解码器decoder，用来屏蔽后面的信息
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-60000)
        return functional.softmax(X.reshape(shape), dim=-1)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, index):
        return self.data[index]

    def reset(self):
        self.data = [0.0] * len(self.data)


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    Defined in :numref:`sec_use_gpu`"""

    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Channels:

    def AWGN(self, Tx_sig, n_var, device='cuda:0'):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)  # 该方案SNR在30dB左右
        return Rx_sig

    # 定义加性高斯白噪声 函数
    def add_awgn(self, y, snr, device='cuda:0'):  # 该方案SNR可指定
        snr1 = 10 ** (snr / 10.0)
        xpower = torch.sum(y ** 2) / y.numel()
        npower = xpower / snr1
        y_noise = torch.randn(size=y.shape).to(device) * torch.sqrt(npower) + y
        return y_noise


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] += theta / norm


def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x


def check_snr(enc_output, voice):
    Ps = (torch.linalg.norm(enc_output - enc_output.mean())) ** 2  # signal power
    Pn = (torch.linalg.norm(enc_output - voice)) ** 2  # noise power
    return 10 * torch.log10(Ps / Pn)
