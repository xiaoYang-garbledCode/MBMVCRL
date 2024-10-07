# 作者: Peach Yang
# 2023年03月21日21时56分44秒
class Config(object):
    def __init__(self):
        self.n_fft = 128
        self.stft_conv1_input = 18
        self.stft_conv1_output = 24
        self.stft_conv2_input = 24
        self.stft_conv2_output = 32
        self.stft_conv3_input = 32
        self.stft_conv3_output = 64
        self.stft_conv4_input = 64
        self.stft_conv4_output = 128


class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        #128
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 24

        # training configs
        self.num_epoch = 40
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128
        self.lambda1 = 1
        self.lambda2 = 0.7
        self.lambda3 = 1
        self.lambda4 = 0.7

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        # new add
        self.num = 11488
        self.name = 'Epilepsy'
        self.n_fft = 64
        self.stft_conv1_input = 2
        self.stft_conv1_output = 16
        self.stft_conv2_input = 16
        self.stft_conv2_output = 32
        self.stft_conv3_input = 32
        self.stft_conv3_output = 64
        self.stft_conv4_input = 64
        self.stft_conv4_output = 128
        self.hop_length = 4

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10
