class Config(object):
    def __init__(self):
        # model configs
        # ckp_last_sleepEDF - 128 - 114 - s3 - m3 - bs - 64.pt
        self.input_channels = 1
        self.block1_output = 32
        self.block2_input = 32
        self.block2_output = 64
        self.block3_input = 64
        self.final_out_channels = 128
        self.kernel_size1 = 20
        self.kernel_size2 = 25
        self.kernel_size3 = 30
        self.stride = 3
        self.maxpooling_stride = 3
        self.num_classes = 5
        self.dropout = 0.35
        self.features_len = 114

        # training configs
        self.num_epoch = 40
        self.batch_size = 64

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.lambda1 = 1
        self.lambda2 = 0.7
        self.lambda3 = 0.001
        self.lambda4 = 0.0007
        self.name = 'sleepEDF'
        # data parameters
        self.drop_last = True

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        # new add
        self.n_fft = 256
        self.stft_conv1_input = 2
        self.stft_conv1_output = 24
        self.stft_conv2_input = 24
        self.stft_conv2_output = 32
        self.stft_conv3_input = 32
        self.stft_conv3_output = 64
        self.stft_conv4_input = 64
        self.stft_conv4_output = 128
        self.hop_length = 8


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50
