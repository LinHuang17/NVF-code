import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Experiment launch: Logistic/Datasets related
        g_logistic = parser.add_argument_group('Logistic')
        g_logistic.add_argument('--exp_id', type=str, default='cs_nvf_run_eval',help='')
        g_logistic.add_argument('--work_base_path', type=str, default='/data1/lin/nvf_results/runs',help='')

        g_logistic.add_argument('--dataset', type=str, default='fh',help='fh | ho3dv3 | ge | comp | obman | dexycb | fphab | stb')
        g_logistic.add_argument('--train_data', type=str, default=['fh'], help='fh | ho3dv3 | ge | comp | obman | dexycb | fphab | stb')
        g_logistic.add_argument('--eval_data', type=str, default=['fh'], help='fh | ho3dv3 | ge | comp | obman | dexycb | fphab | stb')
        
        g_logistic.add_argument('--ds_fh_dir', type=str, default='/data2/lin/FreiHAND/FreiHAND_pub_v2', help='')
        g_logistic.add_argument('--ds_fh_eval_dir', type=str, default='/data2/lin/FreiHAND/FreiHAND_pub_v2_eval', help='')
        g_logistic.add_argument('--ds_comp_dir', type=str, default='/data2/lin/complement', help='')

        g_logistic.add_argument('--wks_size', type=int, default=[500, 500, 1050], help='size of workspace/mm')
        g_logistic.add_argument('--wks_z_shift', type=int, default=675, help='shift of workspace/mm')
        g_logistic.add_argument('--test_wks_size', type=int, default=[500, 500, 1050], help='size of workspace/mm')
        g_logistic.add_argument('--test_wks_z_shift', type=int, default=675, help='shift of workspace/mm')
        g_logistic.add_argument('--is_sampling', type=bool, default=True, help='')
        g_logistic.add_argument('--sample_ratio', type=int, default=20, help='20 | 24 | 16 | 32 for surf')
        g_logistic.add_argument('--bbx_size', type=int, default=[340,340,340], help='size of object bounding box/mm') # 380
        g_logistic.add_argument('--bbx_shift', type=int, default=0, help='shift of object bounding box/mm')
        g_logistic.add_argument('--use_remap', type=bool, default=True, help='')
        g_logistic.add_argument('--use_shape', type=bool, default=False, help='')
        g_logistic.add_argument('--use_scale', type=bool, default=True, help='')
        g_logistic.add_argument('--rdist_norm', type=str, default='uvf', help='normlization method for ray distance, uvf | minmax')

        g_logistic.add_argument('--img_size', type=int, default=[224,224], help='image shape: [224,224] | [128,128]')
        g_logistic.add_argument('--num_views', type=int, default=1, help='How many views to use for multiview network.')

        g_logistic.add_argument('--GPU_ID', default=[0], type=int, help='# of GPUs')
        g_logistic.add_argument('--deterministic', type=bool, default=False, help='')
        g_logistic.add_argument('--seed', type=int, default=0)

        g_logistic.add_argument('--continue_train', type=bool, default=False, help='continue training: load model')
        g_logistic.add_argument('--resume_epoch', type=int, default=0, help='epoch resuming the training')
        g_logistic.add_argument('--eval_perf', type=bool, default=False, help='evaluation: load model')
        g_logistic.add_argument('--eval_epoch', type=int, default=0, help='epoch for eval.')

        g_logistic.add_argument('--load_netG_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        g_logistic.add_argument('--load_optG_checkpoint_path', type=str, default=None, help='path to save checkpoints')

        g_logistic.add_argument('--name', type=str, default='example',
                           help='name of the experiment. It decides where to store/load samples and models')

        # g_logistic.add_argument('--debug_path', type=str, default='/data1/lin/results/NVF/data/dexycb/test', help='')
        
        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--sigma_ratio', type=float, default=0.6, help='perturbation ratio of standard deviation for positions')

        g_sample.add_argument('--num_sample_inout', type=int, default=5000, help='# of sampling points: 5000')

        # MANO related
        g_mano = parser.add_argument_group('MANO')
        g_mano.add_argument('--side', type=str, default='right', help='')
        g_mano.add_argument('--ncomps', type=int, default=45, help='')
        g_mano.add_argument('--use_rot', type=bool, default=True, help='')
        g_mano.add_argument('--use_trans', type=bool, default=False, help='')
        g_mano.add_argument('--center_idx', type=int, default=0, help='0 | 9')
        g_mano.add_argument('--root_idx', type=int, default=9, help='0 | 9')
        g_mano.add_argument('--use_pca', type=bool, default=False, help='')
        g_mano.add_argument('--flat_hand_mean', type=bool, default=False, help='')

        # non-rigid pose related
        g_non_rigid = parser.add_argument_group('non-rigid')
        g_non_rigid.add_argument('--ball_radius', type=int, default=80,  help='radius for ball query')
        g_non_rigid.add_argument('--top_vote_frac', type=float, default=0.5,  help='0.25 | 0.5 | 0.75 | 1.0')

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--batch_size', type=int, default=1, help='input batch size') # 22

        g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')

        g_train.add_argument('--out_type', type=str, default='pysdf', help='rsdf | pysdf | occup | trisdf | klsdf')
        g_train.add_argument('--clamp_dist', type=float, default=5.0, help='')
        g_train.add_argument('--norm_clamp_dist', type=float, default=0.1, help='')
        g_train.add_argument('--eval_clamp_dist', type=float, default=0.02, help='0.1 | 0.06 | 0.02')
        g_train.add_argument('--use_vote', type=bool, default=True, help='')

        g_train.add_argument('--init_type', type=str, default='xavier', help='normal | xavier | kaiming | orthogonal')
        g_train.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal')

        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of stacked layer of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[258, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_vote', nargs='+', default=[258, 1024, 512, 256, 128, 85],
                             type=int, help='# of dimensions of mlp')

        g_model.add_argument('--use_tanh', type=bool, default=True, help='using tanh after last conv of image_filter network')
        g_model.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')

        # Eval. related
        g_eval = parser.add_argument_group('Evaluation')
        g_eval.add_argument('--mesh_step_size', type=int, default=16, help='step size (mm): 4 | 8 | 16 | 24 | 32')
        g_eval.add_argument('--pose_step_size', type=int, default=16, help='step size (mm): 4 | 8 | 16 | 24 | 32')
        g_eval.add_argument('--num_in_batch', type=int, default=150000, help='number of each batch for eval.')
        g_eval.add_argument('--thresh', type=float, default=0.0, help='0.0999 | 0.0 | -0.0999')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
