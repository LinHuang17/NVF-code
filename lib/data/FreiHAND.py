"""
 dataset class of FreiHAND
 for camera-space 3D hand pose estimation in NVF
"""

import os
import sys
import pdb
import random
import logging
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import PIL
import json
import torch
import pickle

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_frustum_utils.data_util import *

from options import BaseOptions


class FreiHAND(Dataset):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, phase='evaluation'):
        self.opt = opt
        # path & state setup
        self.phase = phase

        # 3D->2D projection: 'orthogonal' or 'perspective'
        self.projection_mode = 'perspective'

        # ABBox or Sphere in cam. c.s.
        B_SHIFT = self.opt.bbx_shift
        Bx_SIZE = self.opt.bbx_size[0] // 2
        By_SIZE = self.opt.bbx_size[1] // 2
        Bz_SIZE = self.opt.bbx_size[2] // 2
        self.B_MIN = np.array([-Bx_SIZE, -By_SIZE, -Bz_SIZE])
        self.B_MAX = np.array([Bx_SIZE, By_SIZE, Bz_SIZE])
        # wks box in cam. c.s.
        self.CAM_Bz_SHIFT = self.opt.wks_z_shift
        Cam_Bx_SIZE = self.opt.wks_size[0] // 2
        Cam_By_SIZE = self.opt.wks_size[1] // 2
        Cam_Bz_SIZE = self.opt.wks_size[2] // 2
        self.CAM_B_MIN = np.array([-Cam_Bx_SIZE, -Cam_By_SIZE, -Cam_Bz_SIZE+self.CAM_Bz_SHIFT])
        self.CAM_B_MAX = np.array([Cam_Bx_SIZE, Cam_By_SIZE, Cam_Bz_SIZE+self.CAM_Bz_SHIFT])
        # test wks box in cam. c.s.
        self.TEST_CAM_Bz_SHIFT = self.opt.test_wks_z_shift
        Test_Cam_Bx_SIZE = self.opt.test_wks_size[0] // 2
        Test_Cam_By_SIZE = self.opt.test_wks_size[1] // 2
        Test_Cam_Bz_SIZE = self.opt.test_wks_size[2] // 2
        self.TEST_CAM_B_MIN = np.array([-Test_Cam_Bx_SIZE, -Test_Cam_By_SIZE, -Test_Cam_Bz_SIZE+self.TEST_CAM_Bz_SHIFT])
        self.TEST_CAM_B_MAX = np.array([Test_Cam_Bx_SIZE, Test_Cam_By_SIZE, Test_Cam_Bz_SIZE+self.TEST_CAM_Bz_SHIFT])

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            # transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load annotations
        self.db_data_anno = tuple(load_eval_db_annotation(self.opt.ds_fh_eval_dir, None, self.phase))


    def __len__(self):

        return len(self.db_data_anno)

    def get_img_cam(self, frame_id):

        # shape (H, W, C)/(224, 224, 3)
        rgb_path = read_img_path(frame_id, self.opt.ds_fh_dir, self.phase)
        render = Image.open(rgb_path).convert('RGB')
        w, h = render.size

        # camera intrinsics, mano-related parameters, 3D GT hand joint locations
        cam_K, mano_param, hand_joint_xyz = self.db_data_anno[frame_id]
        cam_K, mano_param, hand_joint_xyz = [np.array(x) for x in [cam_K, mano_param, hand_joint_xyz]]
        # mano pose with glob_rot in axis-angle, mano shape, uv of hand root joint, scale for weak pers.
        # (1, 48), (1, 10), (1, 2), (1, 1) -> (1, 61)
        pose_w_cam_R, shape, hand_root_uv, weak_pers_scale = split_theta(mano_param)
        focal, ppt = get_focal_ppt(cam_K)
        hand_root_xyz = recover_root(hand_root_uv, weak_pers_scale, focal, ppt)

        # original camera intrinsic
        K = np.array(cam_K).reshape(3, 3)
        camera = dict(K=K.astype(np.float32), aug_K=np.copy(K.astype(np.float32)), resolution=(w, h))

        aug_intrinsic = camera['aug_K']
        aug_intrinsic = np.concatenate([aug_intrinsic, np.array([0, 0, 0]).reshape(3, 1)], 1)
        aug_intrinsic = np.concatenate([aug_intrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        aug_intrinsic = torch.Tensor(aug_intrinsic).float()

        # remapping to reference camera
        if self.opt.use_remap:
            curr_cam_mtx = (aug_intrinsic[:3, :3]).numpy()
            ref_cam_mtx = [480.0000, 0.0, 112.0000, 0.0, 480.0000, 112.0000, 0.0, 0.0, 1.0]
            ref_cam_mtx = np.array(ref_cam_mtx).reshape(3, 3)
            # mapx, mapy = cv2.initUndistortRectifyMap(curr_cam_mtx, None, None, ref_cam_mtx, (w,h), 5)
            mapx, mapy = cv2.initUndistortRectifyMap(curr_cam_mtx, None, None, ref_cam_mtx, (w,h), cv2.CV_32FC1)
            render = cv2.remap(np.array(render), mapx, mapy, cv2.INTER_LINEAR)
            render = Image.fromarray(render)
            # use reference camera intrinsic
            aug_intrinsic = np.concatenate([ref_cam_mtx, np.array([0, 0, 0]).reshape(3, 1)], 1)
            aug_intrinsic = np.concatenate([aug_intrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            aug_intrinsic = torch.Tensor(aug_intrinsic).float()

        # img normalization
        render = self.to_tensor(render)

        # 3D hand joint in camera space
        hand_joint_xyz = hand_joint_xyz * 1000

        # metric length of phalangal proximal bone of the middle finger; mm; joint 9 & joint 10
        hand_scale = np.sqrt(np.sum((hand_joint_xyz[10,:] - hand_joint_xyz[9,:]) ** 2))
        hand_scale = (hand_scale - 15) / 25
        
        hand_joint_xyz = torch.Tensor(hand_joint_xyz.T).float()

        return {'img': render,
                'calib': aug_intrinsic,
                # 'hand_shape': torch.Tensor(np.copy(shape).reshape(10,)).float(), 
                'hand_scale': torch.Tensor(np.copy(hand_scale).reshape(1,)).float(), 
                'joint_cam': hand_joint_xyz,
               }


    def get_item(self, index):

        res = {
            'b_min': self.CAM_B_MIN,
            'b_max': self.CAM_B_MAX,
            'test_b_min': self.TEST_CAM_B_MIN,
            'test_b_max': self.TEST_CAM_B_MAX,
        }

        render_data = self.get_img_cam(index)
        res.update(render_data)
        
        return res

    def __getitem__(self, index):
        return self.get_item(index)


if __name__ == '__main__':
    """Test the dataset
    """
    phase = 'training' # training, evaluation
    opt = BaseOptions().parse()
    dataset = FreiHAND(opt, phase=phase)
    print(f'len. of dataset {len(dataset)}')
    # os.makedirs(opt.debug_path, exist_ok=True)