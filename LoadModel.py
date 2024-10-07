# 作者: Peach Yang
# 2023年01月08日21时06分02秒
import torch
# import xlwt  # 需要的模块
# from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from SaveFeatures import save_features
from models.TC import TC
# from utils import _calc_metrics, copy_Files
from config_files import sleepEDF_Configs
from config_files import dataset2a_Configs
from config_files import Epilepsy_Configs
from models.model import base_Model

# dataset2a_Configs  sleepEDF_Configs  Epilepsy_Configs
configs = sleepEDF_Configs.Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dataset2a Epilepsy sleepEDF
data_path = f"./data/{configs.name}".format(configs.name)
training_mode = "self_supervised"
model_weigh = 'ckp_last_' + configs.name + '.pt'
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
model = base_Model(configs).to(device)
chkpoint = torch.load('./saved_models/' + model_weigh)
# 加载模型
model.load_state_dict(chkpoint["model_state_dict"])

# 保存特征
save_features(model, train_dl, device, configs)
