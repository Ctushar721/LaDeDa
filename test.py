import copy
import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
from options.test_options import TestOptions
from validate import validate
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
    vals_array = Testopt.vals.split(',')
    multiclass = Testopt.multiclass.split(',')
    print(f'vals_array: {vals_array}')
    for v_id, val in enumerate(vals_array):
        print(f"eval on {val}")
        print('Testopt.dataroot', Testopt.dataroot)
        opts = copy.deepcopy(Testopt)
        opts.dataroot = '{}/{}'.format(opts.dataroot, val)
        print('opts.dataroot', opts.dataroot)
        opts.classes = os.listdir(opts.dataroot) if multiclass[v_id] else ['']
        opts.no_resize = False
        opts.no_crop = True
        opts.is_aug = False
        acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, opts)
        accs.append(acc)
        aps.append(ap)
        print("({} {:10}) acc: {:.1f}; ap: {:.1f};".format(v_id, val, acc * 100, ap * 100))

    print(f"Mean: acc: {np.array(accs).mean() * 100}, Mean: ap: {np.array(aps).mean() * 100}")


def get_model(model_path, features_dim):
    print("Testopt.preprocess is:", Testopt.preprocess)
    model = LaDeDa9(preprocess_type=Testopt.preprocess, num_classes=1)
    model.fc = torch.nn.Linear(features_dim, 1)
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
    model = get_model(Testopt.model_path, features_dim=Testopt.features_dim)
    test_model(model)
