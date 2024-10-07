
import torch.nn as nn
import torch


class ResBlock(nn.Module):
    """
    A Simple ResNet Block with 2 convolutional layers and a skip connection
    一个简单的 ResNet 块，具有 2个卷积层和一个跳过连接
    """

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
        """ Forward pass of the block."""

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
    """ An encoder for the spectrogram model.
    频谱图模型的编码器
    """

    def __init__(self, n_dim: int) -> None:
        super(CNNEncoder2D_SLEEP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(6, 8, 2, True, False)
        self.conv3 = ResBlock(8, 16, 2, True, True)
        self.conv4 = ResBlock(16, 32, 2, True, True)
        self.n_dim = n_dim

        self.fc = nn.Sequential(
            nn.Linear(128, self.n_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.n_dim, self.n_dim, bias=True),
        )

    def torch_stft(self, X_train: torch.Tensor) -> torch.Tensor:
        """ Compute the STFT of the input."""

        signal = []
        # X_train = [32,1,3000]
        for s in range(X_train.shape[1]):
            # 输入为 shape=(batchsize,L)
            # 输出为 (batchsize,N,T,2)
               # N = n_fft//2 + 1 = 256//2 + 1 =129
            # T= （L-1）// hopsize + 1   hopsize默认为1/4的n_fft)
            # T = （3000 - 1 ） // 64 + 1 = 47
            # (batchsize,N,T,2)
            # spectral.shape = [32,129,43,2]
            spectral = torch.stft(
                X_train[:, s, :],
                n_fft=256,
                hop_length=256 * 1 // 4,
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
        """ Forward pass of the block."""
        # x: [32,1,3000]
        x = self.torch_stft(x)
        # x = self.torch_stft(x)之后的结果为 [32,2,129,43]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x
