import sys
import torch

print(sys.path)

list = []
# features = x.shape[2]
# list.append(x)
# y_test = torch.stack(list)
# y_test1 = torch.stack(list)[:, :, :, :, 0]
# y1 = torch.stack(list)[:, :, :, :, 0].permute(1, 0, 2, 3)
# y2 = torch.stack(list)[:, :, :, :, 1].permute(1, 0, 2, 3)
#
# abs_y1 = torch.log(torch.abs(y1) + 1e-8)
# abs_y2 = torch.log(torch.abs(y2) + 1e-8)
# y = torch.cat([abs_y1, abs_y2], dim=1)
# yx = torch.cat((abs_y1, abs_y2), dim=1)
# print()
#
# signal = []
# # 返回（batch_size，N，T）
# # N = n_fft /2 + 1
# # center=False, 其中win_length(int) 代表窗口大小，默认等于 n_fft。
# # T=[seq_length - win_length + hop_length) ]  / hop_length      hop_length=32 * 1 // 4
# X_train = torch.randn((128, 1,3000))
# for s in range(X_train.shape[1]):
#     spectral = torch.stft(
#         X_train[:, s, :],
#         n_fft=128,
#         hop_length=128 * 1 // 4,
#         center=False,
#         onesided=True,
#         return_complex=False,
#     )
#     signal.append(spectral)
#
# signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
# signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)
#
# k = torch.cat(
#     [
#         torch.log(torch.abs(signal1) + 1e-8),
#         torch.log(torch.abs(signal2) + 1e-8),
#     ],
#     # [32,2,129,43]
#     dim=1,
# )
