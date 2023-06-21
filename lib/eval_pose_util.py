"""
 evaluation for:
    query-to-surface sdf, query-to-joint offset, query-to-joint voting
"""

import os
import json
import time
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

from .geometry import *

# from skimage import measure
from lib.non_rigid_pose_utils import *
from lib.non_rigid_pose_eval_utils import *
from .sdf import create_grid, eval_sdf_vote_grid_frustum


def out_of_plane_mask_calc(cam_pts, calib, img_size):
    # deal with out-of-plane cases
    c2i_rot = calib[:3, :3]
    c2i_trans = calib[:3, 3:4]
    img_sample_pts = torch.addmm(c2i_trans, c2i_rot, torch.Tensor(cam_pts.T).float())
    img_sample_uvs = img_sample_pts[:2, :] / img_sample_pts[2:3, :]

    # normalize to [-1,1]
    transforms = torch.zeros([2,3])
    transforms[0,0] = 1 / (img_size[0] // 2)
    transforms[1,1] = 1 / (img_size[1] // 2)
    transforms[0,2] = -1
    transforms[1,2] = -1
    scale = transforms[:2, :2]
    shift = transforms[:2, 2:3]
    img_sample_norm_uvs = torch.addmm(shift, scale, img_sample_uvs)
    in_img = (img_sample_norm_uvs[0,:] >= -1.0) & (img_sample_norm_uvs[0,:] <= 1.0) & (img_sample_norm_uvs[1,:] >= -1.0) & (img_sample_norm_uvs[1,:] <= 1.0)
    not_in_img = torch.logical_not(in_img).numpy()

    return not_in_img


def eval_pose(opt, net, test_data_loader, save_json_path):

    with torch.no_grad():
        time_arr = []
        joint_xyz_errors = []
        pa_joint_xyz_errors = []
        xyz_pred_list, verts_pred_list = list(), list()
        for test_idx, test_data in enumerate(tqdm(test_data_loader)):
            
            pose_and_err = {}
            # retrieve the data
            resolution_X = int(opt.test_wks_size[0] / opt.pose_step_size)
            resolution_Y = int(opt.test_wks_size[1] / opt.pose_step_size)
            resolution_Z = int(opt.test_wks_size[2] / opt.pose_step_size)
            image_tensor = test_data['img'].cuda()
            calib_tensor = test_data['calib'].cuda()
            joint_tensor = test_data['joint_cam'].cuda()
            if opt.use_shape:
                shape_tensor = test_data['hand_shape'].cuda()
            elif opt.use_scale:
                shape_tensor = test_data['hand_scale'].cuda()
            else:
                shape_tensor = None

            # get all 3D queries
            # create a grid by resolution
            # and transforming matrix for grid coordinates to real world xyz
            b_min = np.array(test_data['test_b_min'][0])
            b_max = np.array(test_data['test_b_max'][0])
            coords, mat = create_grid(resolution_X, resolution_Y, resolution_Z, b_min, b_max, transform=None)
            # (M=KxKxK, 3)
            coords = coords.reshape([3, -1]).T
            # (M,)
            coords_not_in_img = out_of_plane_mask_calc(coords, test_data['calib'][0], opt.img_size)
            # (M,)
            coords_in_img = np.logical_not(coords_not_in_img)
            # (3, N)
            coords_in_frustum = coords[coords_in_img].T

            # transform for proj.
            transforms = torch.zeros([1,2,3]).cuda()
            transforms[:, 0,0] = 1 / (opt.img_size[0] // 2)
            transforms[:, 1,1] = 1 / (opt.img_size[1] // 2)
            transforms[:, 0,2] = -1
            transforms[:, 1,2] = -1

            eval_start_time = time.time()
            # get 2D feat. maps
            net.filter(image_tensor)
            # Then we define the lambda function for cell evaluation
            def eval_func(points):
                points = np.expand_dims(points, axis=0)
                # points = np.repeat(points, net.num_views, axis=0)
                samples = torch.from_numpy(points).cuda().float()

                transforms = torch.zeros([1,2,3]).cuda()
                transforms[:, 0,0] = 1 / (opt.img_size[0] // 2)
                transforms[:, 1,1] = 1 / (opt.img_size[1] // 2)
                transforms[:, 0,2] = -1
                transforms[:, 1,2] = -1
                net.query(samples, calib_tensor, shape_tensor, transforms=transforms)
                # shape (B, 1, N) -> (N)
                eval_sdfs = net.preds[0][0]
                # shape (B, 21, 4, N) -> (21, 4, N)
                eval_votes = net.votes[0]
                return eval_sdfs.detach().cpu().numpy(), eval_votes.detach().cpu().numpy()
            # (N), (84, N), all the predicted query-to-surface sdfs, query-to-joint offsets
            pred_sdfs, pred_votes = eval_sdf_vote_grid_frustum(coords_in_frustum, eval_func, num_samples=opt.num_in_batch)
            # based on estimated mask (queries near surface)
            # (N)
            # pos_anchor_mask = (abs(pred_sdfs) < opt.norm_clamp_dist)
            pos_anchor_mask = (abs(pred_sdfs) < opt.eval_clamp_dist)
            # (3, M)
            est_pts = coords_in_frustum[:, pos_anchor_mask]
            # (21, 4, M)
            est_votes = pred_votes[:, :, pos_anchor_mask]

            # (1, 21, 3), estimated hand pose as hand joint locations via query-to-joint voting
            if opt.top_vote_frac == 1.0:
                est_joint_xyz = recover_joint_xyz_ball_all(est_total_heatmap = torch.Tensor(est_votes).unsqueeze(0).cuda(),
                                                           points = torch.Tensor(est_pts).unsqueeze(0).cuda(),
                                                           volume_length = torch.ones(1, 1).cuda(),
                                                           ball_radius = opt.ball_radius,
                                                          )
            else:
                final_candidate_num = int(pos_anchor_mask.sum() * opt.top_vote_frac)
                est_joint_xyz = recover_joint_xyz_ball(est_total_heatmap = torch.Tensor(est_votes).unsqueeze(0).cuda(),
                                                       points = torch.Tensor(est_pts).unsqueeze(0).cuda(),
                                                       volume_length = torch.ones(1, 1).cuda(),
                                                       ball_radius = opt.ball_radius,
                                                       candidate_num = final_candidate_num,
                                                       )
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            time_arr.append(eval_time)

            # gt joint, est joint, PA est joint
            joint_xyz_gt = np.copy(joint_tensor[0].cpu().numpy().T)
            joint_xyz_est = np.copy(est_joint_xyz.squeeze(0).cpu().numpy())
            joint_xyz_align = rigid_align(joint_xyz_est, joint_xyz_gt)
            
            # error metric
            joint_xyz_errors.append(np.sqrt(np.sum((joint_xyz_est - joint_xyz_gt) ** 2, axis=1)))
            pa_joint_xyz_errors.append(np.sqrt(np.sum((joint_xyz_gt - joint_xyz_align) ** 2, axis=1)))

            # prepare json
            xyz_pred_list.append(joint_xyz_est / 1000)
            verts_pred_list.append(np.zeros((778,3)))
    
    # get error
    mpjpe = np.array(joint_xyz_errors).mean()
    pa_mpjpe = np.array(pa_joint_xyz_errors).mean()
    ave_time = np.array(time_arr).mean()
    
    # dump results
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]
    # save to a json
    with open(save_json_path, 'w') as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)

    return mpjpe, pa_mpjpe, ave_time