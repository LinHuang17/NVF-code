"""
 NVF:
    eval. for camera-space 3D hand pose estimation
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import cv2
import json
import time
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

from lib.data import *
from lib.model import *
from lib.net_util import *
from lib.eval_pose_util import *
from lib.options import BaseOptions

# get options
opt = BaseOptions().parse()


def evaluate(opt):

    # seed
    if opt.deterministic:
        seed = opt.seed
        print("Set manual random Seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmarking enabled")

    # set path
    work_path = os.path.join(opt.work_base_path, f"{opt.exp_id}")
    os.makedirs(work_path, exist_ok=True)
    checkpoints_path = os.path.join(work_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    results_path = os.path.join(work_path, "results")
    os.makedirs(results_path, exist_ok=True)
    tb_dir = os.path.join(work_path, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    tb_runs_dir = os.path.join(tb_dir, "runs")
    os.makedirs(tb_runs_dir, exist_ok=True)
    debug_dir = os.path.join(work_path, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # set gpu environment
    devices_ids = opt.GPU_ID
    num_GPU = len(devices_ids)
    torch.cuda.set_device(devices_ids[0])

    # dataset
    test_dataset_list = []
    test_data_ids = opt.eval_data
    for data_id in test_data_ids:
        if data_id == 'fh':
            test_dataset_list.append(FreiHAND(opt, phase='evaluation'))
        if data_id == 'comp':
            test_dataset_list.append(complement(opt, phase='test'))
    projection_mode = test_dataset_list[0].projection_mode
    test_dataset = ConcatDataset(test_dataset_list)
    # create test data loader
    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  num_workers=opt.num_threads, pin_memory=(opt.num_threads == 0))
                                #   persistent_workers=(opt.num_threads > 0))
                                #   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_dataset))

    # define model, multi-gpu, checkpoint
    netG = HGPIFuNet(opt, projection_mode)
    print('Using Network: ', netG.name)

    def set_eval():
        netG.eval()

    # load checkpoints
    if opt.eval_perf:
        print('Loading for net G ...', opt.load_netG_checkpoint_path)
        netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=torch.device('cpu')))

    # Data Parallel
    # if num_GPU > 1:
    netG = torch.nn.DataParallel(netG, device_ids=devices_ids, output_device=devices_ids[0])
    # netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=devices_ids, output_device=devices_ids[0])
    print(f'Data Paralleling on GPU: {devices_ids}')
    netG.cuda()

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs('%s/%s' % (checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (results_path, opt.name), exist_ok=True)
    opt_log = os.path.join(results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))
    
    # evaluation
    with torch.no_grad():
        set_eval()
        print('evaluate for 3D hand pose (test) ...')
        save_json_path = os.path.join(results_path, opt.name, f'nvf-cam-hand-pose.json')
        eval_hand_pose_error = eval_pose(opt, netG.module, test_data_loader, save_json_path)
        print('eval hand error, mpjpe: {0:06f}, pa_mpjpe: {1:06f}, time: {2:06f}'.format(*eval_hand_pose_error))

if __name__ == '__main__':
    evaluate(opt)
