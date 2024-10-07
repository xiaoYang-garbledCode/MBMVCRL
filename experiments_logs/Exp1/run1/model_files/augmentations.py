import numpy as np
import torch


def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg),
                        config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    # 产生维度为[x.shape[0], x.shape[2]]的数组 [7360, 178]
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        # 将multiply x与产生的factor相乘，之后在中间增加一维，变回[7360,1,178]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    # 产生数组：array([1, 1, 4, ..., 1, 4, 1]) 数的范围[1,4] 数量：7360
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    # 产生于x维度一样的 0数组
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                # x.shape[2]：取数据X的第二个维度的值值为178   num_segs[i] - 1 的范围是[0,3]
                # split_points代表产生长度范围为[1,3]的数组，数组的值的范围为[0,176]
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                # np.split 表示按照split_points这个数组中的值，进行切分orig_steps数组
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # np.random.permutation 将多个数组的顺序打乱，np.concatenate将打乱后的多个数组拼为一个数组,ravel()将这个数组拉成一维数组
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

