# Denis
# coding:UTF-8
import torch

# 给数据加指定SNR的高斯噪声
signal = torch.normal(0, 1, size=(128, 30, 128))
SNR = 12
noise = torch.randn(size=signal.shape)  # 产生N(0,1)噪声数据
noise = noise - torch.mean(noise)  # 均值为0
signal_power = torch.linalg.norm(signal - signal.mean()) ** 2 / signal.numel()  # 此处是信号的std**2
noise_variance = signal_power / torch.pow(torch.tensor(10),torch.tensor((SNR / 10)))  # 此处是噪声的std**2
noise = (torch.sqrt(noise_variance) / torch.std(noise)) * noise  ##此处是噪声的std**2
signal_noise = noise + signal

# signal_noise = signal + torch.normal(0, 0.1, size=signal.shape)

# snr1 = 10 ** (SNR / 10.0)
# xpower = torch.sum(signal ** 2) / signal.numel()
# npower = xpower / snr1
# signal_noise = torch.randn(size=signal.shape) * torch.sqrt(npower) + signal

Ps = (torch.linalg.norm(signal - signal.mean())) ** 2  # signal power
Pn = (torch.linalg.norm(signal - signal_noise)) ** 2  # noise power
snr = 10 * torch.log10(Ps / Pn)
print(snr)
