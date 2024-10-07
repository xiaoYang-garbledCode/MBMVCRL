from torch import nn
import sys
sys.path.append('D://Users//ygj//TCC_final - concat - 副本//models')
from spectrogram_model import CNNEncoder2D_SLEEP
#  spect_model_optimizer = torch.optim.Adam(spect_model.parameters(),
# optimizer got an empty parameter list

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
        # self.spect_model = CNNEncoder2D_SLEEP(256)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        # spect_feats = self.spect_model(x_in)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        # spect_feats
        return logits, x


class SpectModel(nn.Module):
    def __int__(self):
        super(SpectModel).__int__()
        self.spect_model = CNNEncoder2D_SLEEP(256)
        self.feature_spect_feats = 256

    def forward(self, x):
        spect_feats = self.spect_model(x)
        return spect_feats
