import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer



class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        # configs.TC.timesteps = 10
        self.timestep = configs.TC.timesteps
        # ModuleList中存有10个全连接层 (0): Linear(in_features=100, out_features=128, bias=True)
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=0)
        self.device = device
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        # features are (batch_size, #channels, seq_len)     batch_size = 128
        z_aug1 = features_aug1
        # [128, 128, 24]  seq_len =24
        seq_len = z_aug1.shape[2]
        # z_aug1 = [128, 24, 128]
        z_aug1 = z_aug1.transpose(1, 2)
        # z_aug2 = [128, 24, 128]
        z_aug2 = features_aug2

        z_aug2 = z_aug2.transpose(1, 2)
        # batch = batch_size = 128
        batch = z_aug1.shape[0]
        # 随机选取时间戳 torch.randint在[0,166]之间随机产生一个数  如 tensor([105])
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)

        # 时间步长和批次的平均值
        nce = 0
        # self.num_channels = configs.final_out_channels
        # encode_samples = torch.empty((10, 128, 128)).float().to(self.device)   [[[ 0,0,0...]]] 零数组
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(1, self.timestep + 1):
            # z_aug2 ： [128, 24, 128]                         # batch = 128  num_channels = 128
            # z_aug2[:, t_samples + i, :] = [128, 128]
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        # encode_samples = [10, 128, 128]

        # forward_seq = [128, 4, 128]  其中t_samplets = 3
        forward_seq = z_aug1[:, :t_samples + 1, :]
        # c_t 与z_aug1有关    encode_samples 与与z_aug2有关
        # c_t = [128, 100]
        c_t = self.seq_transformer(forward_seq)
        # pred = torch.empty((10, 128, 128)).float().to(self.device)
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            # WK中存有10个全连接层 (0): Linear(in_features=100, out_features=128, bias=True)
            linear = self.Wk[i]
            # pred[0] = [128,128]  linear的输出也是[128,128]
            pred[i] = linear(c_t)
        # 最终 pred = [10,128,128]
        for i in np.arange(0, self.timestep):
            # encode_samples 与与z_aug2有关
            # encode_samples=[10, 128, 128]
            # torch.mm(a, b) 是矩阵a和b矩阵相乘
            # encode_samples[i] = [128, 128] torch.transpose(pred[i], 0, 1) = [128,128]
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)