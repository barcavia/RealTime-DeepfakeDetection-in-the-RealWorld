import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
from options.test_options import TestOptions
from util import Logger
from validate import validate
import torchvision
from torchvision import transforms
import random
from networks.LaDeDa import LaDeDa9
from networks.Tiny_LaDeDa import tiny_ladeda
from test_config import *

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"seed: {seed}")

def test_model(model):
    accs, aps = [], []
    for v_id, val in enumerate(vals):
        print(f"eval on {val}")
        Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
        Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
        Testopt.no_resize = False
        Testopt.no_crop = True
        Testopt.is_aug = False
        acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, Testopt)
        accs.append(acc)
        aps.append(ap)
        print("({} {:10}) acc: {:.1f}; ap: {:.1f};".format(v_id, val, acc * 100, ap * 100))

    print(f"Mean: acc: {np.array(accs).mean() * 100}, Mean: ap: {np.array(aps).mean() * 100}")


def get_model(model_path, features_dim):
    model = LaDeDa9(pretrained=False, num_classes=1)
    model.fc = nn.Linear(features_dim, 1)
    from collections import OrderedDict
    from copy import deepcopy
    state_dict = torch.load(model_path, map_location='cpu')
    pretrained_dict = OrderedDict()
    for ki in state_dict.keys():
        pretrained_dict[ki] = deepcopy(state_dict[ki])
    model.load_state_dict(pretrained_dict, strict=True)
    print("model has loaded")
    model.eval()
    model.cuda()
    model.to(0)
    return model

if __name__ == '__main__':
    set_seed(42)
    Testopt = TestOptions().parse(print_options=False)
    # evaluate model
    # LaDeDa's features_dim = 2048
    # Tiny-LaDeDa's features_dim = 8
    model = get_model(model_path, features_dim=Testopt.features_dim)
    test_model(model)
