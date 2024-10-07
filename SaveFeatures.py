# 作者: Peach Yang
# 2023年01月08日21时22分33秒
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from datetime import datetime
from sklearn.linear_model import SGDClassifier


def save_features(model, train_loader, device, config):
    model.train()
    features_total = []
    # total_loss = []
    data_object = str(datetime.now())
    data_object = data_object.replace(' ', '-')
    data_object = data_object.replace('.', '-')
    data_object = data_object.replace(':', '-')
    # loss_txt = 'loss-%s-loss-%s.txt' % (config.name, data_object)
    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        predictions1, features = model(data)
        features = features.cpu().detach().numpy()
        features_total.append(features)

    features_array = np.array(features_total)
    path = './output/%s-%s.npy' % (data_object, config.name)
    np.save(path, features_array)
