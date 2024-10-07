from torch import nn
import torch
import sys

sys.path.append('D://Users//ygj//otherData_train//TCC-otherData-branch-spect//models')
from spectrogram_model import CNNEncoder2D_SLEEP


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        # self.spect_model = CNNEncoder2D_SLEEP(256)
        self.batch_size = configs.batch_size
        self.conv_branch1_block_1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.block1_output, kernel_size=configs.kernel_size1,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size1 // 2)),
            nn.BatchNorm1d(configs.block1_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_branch1_block_2 = nn.Sequential(
            nn.Conv1d(configs.block2_input, configs.block2_output, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.block2_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1)
        )

        self.conv_branch1_block_3 = nn.Sequential(
            nn.Conv1d(configs.block3_input, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1),
        )

        self.conv_branch2_block_1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.block1_output, kernel_size=configs.kernel_size2,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size2 // 2)),
            nn.BatchNorm1d(configs.block1_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_branch2_block_2 = nn.Sequential(
            nn.Conv1d(configs.block2_input, configs.block2_output, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.block2_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1)
        )

        self.conv_branch2_block_3 = nn.Sequential(
            nn.Conv1d(configs.block3_input, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1),
        )
        self.conv_branch3_block_1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.block1_output, kernel_size=configs.kernel_size3,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size3 // 2)),
            nn.BatchNorm1d(configs.block1_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_branch3_block_2 = nn.Sequential(
            nn.Conv1d(configs.block2_input, configs.block2_output, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.block2_output),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1)
        )

        self.conv_branch3_block_3 = nn.Sequential(
            nn.Conv1d(configs.block3_input, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=configs.maxpooling_stride, padding=1),
        )
        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x1 = self.conv_branch1_block_1(x_in)
        x1 = self.conv_branch1_block_2(x1)
        x1 = self.conv_branch1_block_3(x1)  # dataset2a (128,128,44)    Epilepsy (128, 128, 24)
        x2 = self.conv_branch2_block_1(x_in)
        x2 = self.conv_branch2_block_2(x2)
        x2 = self.conv_branch2_block_3(x2)  # dataset2a (128,128,44) Epilepsy (128, 128, 24)
        x3 = self.conv_branch3_block_1(x_in)
        x3 = self.conv_branch3_block_2(x3)
        x3 = self.conv_branch3_block_3(x3)  # dataset2a (128,128,44) Epilepsy (128, 128, 24)
        # dataset2a batch_size = 128 不好
        # (128, 3072)
        final = torch.cat((x1, x2, x3), dim=2)  # dataset2a(128,16256)  Epilepsy(128, 9216)
        # pFD batch_size = 64 故改 final.reshape(128, 128, -1) 为 final.reshape(64, 128, -1)
        #final = final.reshape(self.batch_size, 128, -1)  # dataset2a (128,128,381)  Epilepsy [128, 128, 72]
        x_flat = final.reshape(final.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, final


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: bool = False,
            pooling: bool = False,
    ) -> None:
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual

        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)

        return out


class CNNEncoder2D_SLEEP(nn.Module):
    def __init__(self, configs) -> None:
        super(CNNEncoder2D_SLEEP, self).__init__()
        self.batch_size = configs.batch_size
        self.conv1 = nn.Sequential(
            # Epilepsy :  nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1)
            nn.Conv2d(configs.stft_conv1_input, configs.stft_conv1_output, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(configs.stft_conv1_output),
            nn.ELU(inplace=True),
        )
        self.n_fft = configs.n_fft
        self.hop_length = configs.hop_length
        #  conv2只有pFD是 最后一个是True
        self.conv2 = ResBlock(configs.stft_conv2_input, configs.stft_conv2_output, 1, True, False)
        self.conv3 = ResBlock(configs.stft_conv3_input, configs.stft_conv3_output, 1, True, True)
        self.conv4 = ResBlock(configs.stft_conv4_input, configs.stft_conv4_output, 1, True, True)

    def torch_stft(self, X_train: torch.Tensor) -> torch.Tensor:
        signal = []
        # 返回（batch_size，N，T）
        # N = n_fft /2 + 1
        # center=False, 其中win_length(int) 代表窗口大小，默认等于 n_fft。
        # T=[seq_length - win_length + hop_length) ]  / hop_length      hop_length=32 * 1 // 4
        for s in range(X_train.shape[1]):
            spectral = torch.stft(
                X_train[:, s, :],
                n_fft=self.n_fft,
                hop_length=self.n_fft * 1 // self.hop_length,
                center=False,
                onesided=True,
                return_complex=False,
            )
            signal.append(spectral)

        signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
        signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

        return torch.cat(
            [
                torch.log(torch.abs(signal1) + 1e-8),
                torch.log(torch.abs(signal2) + 1e-8),
            ],
            # [32,2,129,43]
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.torch_stft(x)  # HAR n_fft = 32（128，18，17，13） n_fft = 24 (128,18,13,18)
        x = self.conv1(x)  # (128,24,17,13)
        x = self.conv2(x)  # (128,32,17,13)
        x = self.conv3(x)  # (128,32,8,6)
        x = self.conv4(x)  # (128,32,4,3)            (128,32,3,4)
        # x = self.fc(x)
        features1 = x.shape[2]
        features2 = x.shape[3]
        # n_fft = 32 Epilepsy = (128, -1, 12)
        x = x.reshape(self.batch_size, -1, features1 * features2)
        return x
