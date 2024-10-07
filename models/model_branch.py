# 作者: Peach Yang
# 2023年03月20日21时12分53秒
from torch import nn
import torch


class base_Branch_Model(nn.Module):
    def __init__(self, configs):
        super(base_Branch_Model, self).__init__()

        self.conv_branch1_block_1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size1,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size1 // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_branch1_block_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_branch1_block_3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.conv_branch2_block_1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size2,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size2 // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_branch2_block_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_branch2_block_3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_branch3_block_1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size3,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size3 // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_branch3_block_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_branch3_block_3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels * 3, configs.num_classes)

    def forward(self, x_in):
        x1 = self.conv_branch1_block_1(x_in)
        x1 = self.conv_branch1_block_2(x1)
        x1 = self.conv_branch1_block_3(x1)  # dataset2a (128,128,44)    Epilepsy (128, 128, 24)
        x2 = self.conv_branch2_block_1(x_in)
        x2 = self.conv_branch2_block_2(x2)
        x2 = self.conv_branch2_block_3(x2)  # dataset2a (128,128,44) Epilepsy (128, 128, 24)
        x3 = self.conv_branch3_block_1(x_in)
        x3 = self.conv_branch3_block_2(x3)
        x3 = self.conv_branch3_block_3(x3)   # dataset2a (128,128,44) Epilepsy (128, 128, 24)
        # dataset2a batch_size = 128 不好
        # (128, 3072)
        x1 = x1.reshape(32, -1)
        x2 = x2.reshape(32, -1)
        x3 = x3.reshape(32, -1)
        final = torch.cat((x1, x2, x3), dim=1)  # dataset2a(128,16256)  Epilepsy(128, 9216)
        final = final.reshape(32, 128, -1)  # dataset2a (128,128,381)  Epilepsy [128, 128, 72]
        x_flat = final.reshape(final.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, final
