"""
 voting utilities:
    based on "Point-to-Point Regression PointNet for 3D Hand Pose Estimation"
"""

import torch

import pdb


def generate_offset_heatmaps_knn_ball(points, gt_joint_xyz, volume_length, knn_K, ball_radius, use_vote_mask, vote_mask=None):
    '''
    Generate masked distance & unit vector field heat-maps
    :param points: [B, 3, N] 3D coordinates of points
    :param gt_joint_xyz: [B, J, 3]
    :param volume_length: [B, 1] 
    :param vote_mask: [B, 1, N]
    :return: total_heatmap[B, J, 4, N] 
    '''
    # (B, J, 3, N)
    offset_vect = gt_joint_xyz.unsqueeze(-1).expand(-1, -1, -1,points.size(-1)) - points.unsqueeze(1).expand(-1, gt_joint_xyz.size(1), -1, -1)
    # (B, J, 3, N)
    offset_diff = torch.mul(offset_vect, offset_vect)
    # (B, J, N)
    offset_diff = torch.sqrt(offset_diff.sum(2))
    # (B, J, N)
    offset_diff_local_R = torch.mul(offset_diff, volume_length.expand(-1, offset_diff.size(1)).unsqueeze(-1).expand(-1, -1, offset_diff.size(-1)))
    
    if use_vote_mask:
        # (B, J, N) = (B, J, N) + (B, 1, N)
        offset_diff_local_R = offset_diff_local_R + (torch.logical_not(vote_mask).float() * ball_radius * 2)
    # (B, J, N), 3D distance heatmap
    dist_map = torch.max(ball_radius - offset_diff_local_R, torch.zeros_like(offset_diff)) / ball_radius

    # (B, J, 3, N), 3D unit vector fields
    offset_diff = torch.max(offset_diff, torch.ones_like(offset_diff) * 1e-10).unsqueeze(2).expand(-1, -1, 3, -1)
    # (B, J, 3, N)
    unit_offset_map = torch.div(offset_vect, offset_diff)
    unit_offset_map[dist_map.eq(0).unsqueeze(2).expand(-1, -1, 3, -1)] = 0

    # (B, J, 4, N)
    total_heatmap = torch.cat((dist_map.unsqueeze(2), unit_offset_map), 2)
    
    # nn_idx: (B, J, (N-knn_K))
    _, nn_idx = torch.topk(dist_map, points.size(-1) - knn_K, 2, largest=False, sorted=False)
    # nn_idx: (B, J, 4, (N-knn_K))
    nn_idx = nn_idx.unsqueeze(2).expand(-1,-1,4,-1) 
    # nn_idx: (B, J, 4, N)
    total_heatmap.scatter_(-1, nn_idx, 0.0) 
    
    return total_heatmap


def recover_joint_xyz_ball_all(est_total_heatmap, points, volume_length, ball_radius):
    '''
    Recover hand joint xyz from heat-maps
    :param est_total_heatmap: [B, J, 4, N] 
    :param points: [B, 3, N] 
    :param volume_length: [B, 1] 
    :return: [B, J, 3] 
    '''
    # recover offset_vect from unit map
    # (B, J, N)
    offset_diff_local_R = (1.0 - est_total_heatmap[:, :, 0, :]) * ball_radius
    # (B, J, N)
    offset_diff = torch.div(offset_diff_local_R, volume_length.expand(-1, offset_diff_local_R.size(1)).unsqueeze(-1).expand(-1, -1, offset_diff_local_R.size(-1)))
    # (B, J, 3, N)
    offset_diff = offset_diff.unsqueeze(2).expand(-1, -1, 3, -1)
    # (B, J, 3, N)
    offset_vect = torch.mul(est_total_heatmap[:, :, 1:, :], offset_diff)
    
    # compute estimation points from offset_vect
    # (B, J, 3, N)
    est_joint_xyz = points.unsqueeze(1).expand(-1, est_total_heatmap.size(1),-1,-1) + offset_vect

    return weighted_mean_fusion_all(est_joint_xyz, est_total_heatmap[:,:,0,:])


def weighted_mean_fusion_all(est_xyz, weights):
    '''
    Weighted fuction 
    :return: [B, J, 3] 
    '''
    # fusion method: weighted mean
    # (B, J, N)
    weights_norm = torch.sum(weights, -1, keepdim=True).expand(-1, -1, weights.shape[-1])
    # (B, J, N)
    weights = torch.div(weights, weights_norm)
    # (B, J, 3, N)
    weights = weights.unsqueeze(2).expand(-1, -1, 3, -1)
    
    # (B, J, 3, N)
    est_xyz = torch.mul(est_xyz, weights)               
    # (B, J, 3)
    est_xyz = torch.sum(est_xyz, -1)                 
    # (B, J, 3) (unweighted mean)
    #est_xyz = torch.mean(est_xyz, -1) 
    return est_xyz


def recover_joint_xyz_ball(est_total_heatmap, points, volume_length, ball_radius, candidate_num=5):
    '''
    Recover hand joint xyz from heat-maps
    :param est_total_heatmap: [B, J, 4, N] 
    :param points: [B, 3, N] 
    :param volume_length: [B, 1] 
    :return: [B, J, 3] 
    '''
    # recover offset_vect from unit map
    # (B, J, N)
    offset_diff_local_R = (1.0 - est_total_heatmap[:, :, 0, :]) * ball_radius
    # (B, J, N)
    offset_diff = torch.div(offset_diff_local_R, volume_length.expand(-1, offset_diff_local_R.size(1)).unsqueeze(-1).expand(-1, -1, offset_diff_local_R.size(-1)))
    # (B, J, 3, N)
    offset_diff = offset_diff.unsqueeze(2).expand(-1, -1, 3, -1)
    # (B, J, 3, N)
    offset_vect = torch.mul(est_total_heatmap[:, :, 1:, :], offset_diff)
    
    # compute estimation points from offset_vect
    # (B, J, 3, N)
    est_joint_xyz = points.unsqueeze(1).expand(-1, est_total_heatmap.size(1),-1,-1) + offset_vect

    return weighted_mean_fusion(est_joint_xyz, est_total_heatmap[:,:,0,:], candidate_num)


def weighted_mean_fusion(est_joint_xyz, heatmap, candidate_num):
    '''
    Weighted fuction 
    :return: [B, J, 3] 
    '''
    # select candidate points
    # weights: (B, J, candidate_num), nn_idx: (B, J, candidate_num)
    weights, nn_idx = torch.topk(heatmap, candidate_num, 2, largest=True, sorted=False)
    # (B, J, 3, candidate_num)
    nn_idx = nn_idx.unsqueeze(2).expand(-1, -1, 3, -1) 
    # (B, J, 3, candidate_num)
    est_xyz = est_joint_xyz.gather(-1,nn_idx)
    
    # fusion method: weighted mean
    weights_norm = torch.sum(weights, -1, keepdim=True).expand(-1, -1, candidate_num)
    # (B, J, candidate_num)
    weights = torch.div(weights, weights_norm)
    # (B, J, 3, candidate_num)
    weights = weights.unsqueeze(2).expand(-1, -1, 3, -1)
    
    # (B, J, 3, candidate_num)
    est_xyz = torch.mul(est_xyz, weights)               
    # (B, J, 3)
    est_xyz = torch.sum(est_xyz, -1)                 
    # (B, J, 3) (unweighted mean)
    #est_xyz = torch.mean(est_xyz, -1) 
    return est_xyz