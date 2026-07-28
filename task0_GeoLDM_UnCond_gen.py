# Rdkit import should be first, do not move it
import evo

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset, analyze
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
from copy import deepcopy
import random
from qm9.visualizer import plot_data3d,save_xyz_file,load_molecule_xyz
from equivariant_diffusion import utils as diffusion_utils
from contextlib import contextmanager
import matplotlib.pyplot as plt
import utils
import sys
import csv
import os
import numpy as np
import os
# GPU visibility is controlled by scripts/run_suite.sh before torch is imported.
parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='qm9_latent2')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true',
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

# VAE args
parser.add_argument('--latent_nf', type=int, default=1,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=9,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=256,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9_second_half',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', default=True, help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument('--resume', type=str, default='outputs/geoldm_qm9_second_half',
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=100,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda:0'
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)

# Wandb config
mode = 'disabled'
property_norms = None
args.context_node_nf = 0

model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, None)
model = model.to(device)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)



@contextmanager
def suppress_print():
    # 保存原始的标准输出
    original_stdout = sys.stdout
    # 重定向标准输出到空设备
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'), map_location=device)
        model.load_state_dict(flow_state_dict)
        model.eval()

    # model_ema.dynamics.device = device
    print(model.buffer)

    args.n_stability_samples = 26

    NPops = 100
    print("NPops:", NPops)
    args.n_stability_samples = 100
    max_n_nodes = 29
    min_n_nodes = 5
    Obj = ['alpha', 'gap','homo','lumo','mu','Cv']
    preds = evo.Get_Pred(Obj, device)
    alpha = []
    gap = []
    homo = []
    lumo = []
    mu = []
    Cv = []




    for iter in range(1000):
        Pop = evo.InitPop(NPops, nodes_dist, args, device, model, dataset_info, prop_dist, False, min_n_nodes, preds,
                          max_n_nodes, True)
        Pop = evo.EvalPop_Con(Pop, dataset_info)
        Pop = evo.EvalPop_Obj(Pop, device, preds, max_n_nodes)

        alpha_values = [item[Obj[0]].item() for item in Pop if item['Constraint'] == 0]
        gap_values = [item[Obj[1]].item() for item in Pop if item['Constraint'] == 0]
        homo_values = [item[Obj[2]].item() for item in Pop if item['Constraint'] == 0]
        lumo_values = [item[Obj[3]].item() for item in Pop if item['Constraint'] == 0]
        mu_values = [item[Obj[4]].item() for item in Pop if item['Constraint'] == 0]
        Cv_values = [item[Obj[5]].item() for item in Pop if item['Constraint'] == 0]

        alpha.append(deepcopy(alpha_values))
        gap.append(deepcopy(gap_values))
        homo.append(deepcopy(homo_values))
        lumo.append(deepcopy(lumo_values))
        mu.append(deepcopy(mu_values))
        Cv.append(deepcopy(Cv_values))

    values = [alpha,gap,homo,lumo,mu,Cv]
    with open("./EvoResults/task0_half/values.pkl", "wb") as file:
        pickle.dump(values, file)


    def savedualobj(obj1,obj2,obj1name,obj2name):
        plt.figure(figsize=(8, 6))
        for i in range(len(obj1)):
            x = obj1[i]
            y = obj2[i]
            plt.scatter(x, y, alpha=0.3,c='blue')

        # 添加图例
        plt.legend()

        # 添加标题和坐标轴标签
        plt.title(obj1name+obj2name)
        plt.xlabel(obj1name)
        plt.ylabel(obj2name)

        # 显示图形
        plt.grid(True)
        plt.show()
        name = 'EvoResults/task0_half/'+str(obj1name+obj2name)
        # 显示图表
        plt.savefig(name, dpi=300)

    for i in range(6):
        for j in range(6):
            if i!=j:
                savedualobj(values[i],values[j],Obj[i],Obj[j])





if __name__ == "__main__":
    main()
