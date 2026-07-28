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
from evo import InitPop
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
import itertools
import numpy as np
import os
import evis
from MOEA import DEMO
# GPU visibility is controlled by scripts/run_suite.sh before torch is imported.
from demo_runtime.legacy import env_bool
use_EDM = env_bool("DEMO_USE_EDM", default=True)
from MOEA import DEMO

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
if use_EDM:
    args.resume = 'tfgmodels/EDMsecond'
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

if use_EDM:
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, None)
else:
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
    # --- 1. 环境与模型初始化 ---
    evo.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟 args (请根据实际情况替换)
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model_ema.npy'), map_location=device)
        model.load_state_dict(flow_state_dict)
        model.eval()

    # 加载预存的背景样本数据
    with open('./EvoResults/task0_half/values.pkl', 'rb') as f:
        samplevalues = pickle.load(f)
    args.xtb = False
    dataloaders, _ = dataset.retrieve_dataloaders(args)

    MOEA_NAME = 'TOPNUUVR'
    USE_HV_TRIGGER = False
    max_n_nodes, min_n_nodes = 29, 5
    numberoforun = 20  # 重复20次实验

    if use_EDM:
        base_save_root = f'./DEMO/{MOEA_NAME}_MOP_EDM/'
    else:
        base_save_root = f'./DEMO/{MOEA_NAME}_MOP_GEOLDM/'

    # 初始化全能可视化工具
    viz = evis.EvoVisualizer(
        save_dir=base_save_root,
        dataset_info=dataset_info,
        train_data=dataloaders['train'].dataset.data
    )
    tracker = evis.EvoTracker()

    # --- 2. 属性组合循环 ---
    prop = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv']
    # combinations = list(itertools.combinations(prop, 2))
    comb2 = [('alpha', 'gap'), ('gap', 'homo'), ('homo', 'lumo'), ('lumo', 'mu'), ('mu', 'Cv'), ('Cv', 'alpha')]
    comb3 = [('alpha', 'homo', 'mu'), ('gap', 'lumo', 'Cv'), ('alpha', 'lumo', 'Cv'), ('gap', 'homo', 'mu')]
    comb4 = comb2 + comb3
    combinations = comb4
    NPops_numbers = [32]

    for Obj in combinations:
        print(f"\n>>> Start: {Obj}")
        multi_tracker = evis.MultiRunTracker()
        preds = evo.Get_Pred(Obj, device)

        # 准备本组属性的背景样本点
        sample1 = samplevalues[prop.index(Obj[0])]
        sample2 = samplevalues[prop.index(Obj[1])]
        current_samples = [sample1, sample2]
        gens = 20
        if len(Obj) == 3:
            current_samples.append(samplevalues[prop.index(Obj[2])])
            gens = 20

        for NPops in NPops_numbers:
            print(f"  Population Size: {NPops}")

            all_runs_hvs = []
            last_gen_hvs = []

            for indextime in range(numberoforun):
                print(f"    Run {indextime + 1}/{numberoforun}...")
                tracker.reset()

                scheduler = evo.AdaptiveNoiseScheduler(
                    dataset_info=dataset_info,
                    initial_noise=1000,
                    min_noise=0,
                    max_noise=1000,
                    step_size=10,
                    drop_factor=0.5,
                    use_hv_trigger=USE_HV_TRIGGER
                )
                # 初始化种群
                Pop = evo.InitPop(NPops, nodes_dist, args, device, model, dataset_info, prop_dist,
                                  False, min_n_nodes, preds, max_n_nodes, True)
                # 初始评价与记录
                Pop = evo.Get_Fitness_Pareto(Pop, dataset_info, device, preds, max_n_nodes, Obj, None, 0)
                UUVR = evo.get_population_duplicate_rate(Pop, dataset_info, NPops, usecon=False)

                current_noise = scheduler.update(Off=Pop, Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj)
                tracker.update(Pop, Obj, AddNoise=current_noise, score=scheduler.score, UUVR=UUVR)

                # --- 进化主循环 (20代) ---
                for iter_idx in range(gens):
                    Off = evo.InitPop(NPops, nodes_dist, args, device, model, dataset_info, prop_dist,
                                  False, min_n_nodes, preds, max_n_nodes, True)

                    # 评价与精英保留
                    Pop = evo.Get_Fitness_Pareto(Off + Pop, dataset_info, device, preds, max_n_nodes, Obj, None, 0)
                    Pop = evo.EnvironmentalSelectionCon(Pop, NPops, Obj)

                    current_noise = scheduler.update(Off=Off, Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj)
                    current_noise = 1000
                    UUVR = evo.get_population_duplicate_rate(Pop, dataset_info, NPops, usecon=False)

                    tracker.update(Pop, Obj, AddNoise=current_noise, score=scheduler.score, UUVR=UUVR)

                    print(f"  {MOEA_NAME} {use_EDM} | Gen {iter_idx + 1} |  Noise: {current_noise} | HV: {tracker.metrics['HV'][-1]:.4f} | UUVR  ")

                multi_tracker.add('HV', tracker.metrics['HV'])
                multi_tracker.add('AvgAddT', tracker.metrics['AddNoise'])  # 添加到统计器
                multi_tracker.add('Score', tracker.metrics['score'])  # 添加到统计器
                multi_tracker.add('UUVR', tracker.metrics['UUVR'])  # 添加到统计器

                # --- 单轮实验可视化 ---
                viz.plot_evolution(
                    tracker, Obj, interval=2,
                    samples=current_samples,
                    others=f"Run{indextime}",
                    angles=[(45, 45), (0, 90)] if len(Obj) == 3 else [(None, None)]
                )

                # 计算并缓存本轮的 HV
                hv_list = viz.calculate_hv_list(tracker, Obj, minimize=True)
                all_runs_hvs.append(hv_list)
                last_gen_hvs.append(hv_list[-1])

                # --- 20次实验均值统计 ---
                # print(f"  正在计算并保存 {Obj} 的统计结果...")

                # 记录 CSV：最后一代的均值和标准差
            viz.save_stats_csv(f"{MOEA_NAME}_HV_FullLast_HVs", last_gen_hvs, Obj, NPops)

            mean_hv, std_hv = multi_tracker.get_aggregated_stats('HV')
            viz.plot_mean_curve_with_std(mean_hv, std_hv, "Hypervolume",
                                         f"Mean HV over {numberoforun} Runs on {Obj}",
                                         f"MOP_HV_{Obj}_{MOEA_NAME}")

            mean_sc, std_sc = multi_tracker.get_aggregated_stats('Score')
            viz.plot_mean_curve_with_std(mean_sc, std_sc, "Score",
                                         f"Mean score over {numberoforun} Runs on {Obj}",
                                         f"MOP_SCORE_{Obj}_{MOEA_NAME}")

            mean_an, std_an = multi_tracker.get_aggregated_stats('AvgAddT')
            viz.plot_mean_curve_with_std(mean_an, std_an, "AvgAddT",
                                         f"Mean AvgAddT over {numberoforun} Runs on {Obj}",
                                         f"MOP_AN_{Obj}_{MOEA_NAME}")

            mean_UUVR, std_UUVR = multi_tracker.get_aggregated_stats('UUVR')
            viz.plot_mean_curve_with_std(mean_UUVR, mean_UUVR, "UUVR",
                                         f"Mean UUVR over {numberoforun} Runs on {Obj}",
                                         f"MOP_UUVR_{Obj}_{MOEA_NAME}")

            summary_csv_path = os.path.join(base_save_root, f"Aggregate_Statson_{Obj}_{MOEA_NAME}.csv")
            with open(summary_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 更新表头
                writer.writerow(
                    ['Gen', 'Mean_HV', 'Std_HV', 'Mean_AN', 'Std_AN', 'Mean_Score', 'Std_Score', 'Mean_UUVR',
                     'Std_UUVR'])

                # 确保长度一致 (以防万一)
                length = min(len(mean_hv), len(mean_an))
                for i in range(length):
                    writer.writerow([
                        i,
                        mean_hv[i], std_hv[i],
                        mean_an[i], std_an[i],
                        mean_sc[i], std_sc[i],
                        mean_UUVR[i], std_UUVR[i]
                    ])

    # print("\n[FINISH] 所有实验已完成并保存。")




if __name__ == "__main__":
    main()
