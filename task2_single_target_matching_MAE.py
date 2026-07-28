# Rdkit import should be first, do not move it
import evo

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import argparse
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset, analyze
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
from contextlib import contextmanager
import utils
import sys
import csv
import numpy as np
import os
import evis
# GPU visibility is controlled by scripts/run_suite.sh before torch is imported.
use_EDM = False

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
if use_EDM:
    args.resume = 'tfgmodels/EDMsecond'
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

    # --- 1. 配置与初始化 ---
    evo.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model_ema.npy'), map_location=device)
        model.load_state_dict(flow_state_dict)
        model.eval()

    # 加载预存的背景样本数据
    args.xtb = False
    # dataloaders, _ = dataset.retrieve_dataloaders(args)
    MOEA_NAME = "EGD"


    if use_EDM:
        save_path = f'./DEMO/{MOEA_NAME}_EDM_SSOP/'
    else:
        save_path = f'./DEMO/{MOEA_NAME}_GEOLDM_SSOP/'

    viz = evis.EvoVisualizer(save_dir=save_path, dataset_info=dataset_info)

    # 单位换算系数
    conversion = {
        'alpha': 1.0, 'gap': 1000.0, 'homo': 1000.0,
        'lumo': 1000.0, 'mu': 1.0, 'Cv': 1
    }

    prop = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv']
    comb1 = [['alpha'], ['gap'], ['homo'], ['lumo'], ['mu'], ['Cv']]
    comb2 = [('Cv', 'mu'), ('gap', 'mu'), ('alpha', 'mu'), ('homo', 'lumo'), ('lumo', 'mu'), ('lumo', 'gap'), ('homo', 'gap')]

    # combinations = list(itertools.combinations(prop, 1))  # 这里改 1 或 3 都行
    combinations = comb1
    NPops_numbers = [32]
    args.xtb=None

    dataloaders, _ = dataset.retrieve_dataloaders(args)
    train_data = dataloaders['train'].dataset.data
    max_n_nodes, min_n_nodes = 29, 5
    tracker = evis.EvoTracker()
    numberoforun = 20

    for Obj in combinations:
        print(f"\n>>> Start: {Obj}")

        for NPops in NPops_numbers:

            multi_tracker = evis.MultiRunTracker()
            preds = evo.Get_Pred(Obj, device)

            for run_idx in range(numberoforun):
                tracker.reset()
                scheduler = evo.AdaptiveNoiseScheduler(
                    dataset_info=dataset_info,
                    initial_noise=1000,
                    min_noise=0,
                    max_noise=1000,
                    step_size=10,
                    drop_factor=0.5,
                )

                # 1. 采样目标值
                sampled_idx = torch.randint(1, 50000, size=(1,))
                ObjValue = {name: train_data[name][sampled_idx].item() * conversion[name] for name in Obj}

                # 2. 初始化种群
                Pop = evo.InitPop(NPops, nodes_dist, args, device, model, dataset_info, prop_dist, False, min_n_nodes, preds, max_n_nodes,
                                  True)
                Pop = evo.Get_Fitness_multi_dis_MAE_normalize(Pop, dataset_info, device, preds, max_n_nodes, Obj, None, 0,
                                                              ObjValue)

                current_noise = scheduler.update(Off=Pop, Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj, calHV=False)
                tracker.update(Pop, Obj, MMAE=np.mean([item['MAE'] for item in Pop], axis=0), AddNoise=current_noise, score=scheduler.score)
                totalgen = 10
                # 3. 进化迭代
                for gen in range(totalgen):
                    ParentSize = int(NPops / 2)
                    Parent = evo.k_tournament_selection(Pop, 2, ParentSize)
                    for i in range(len(Parent)): Parent[i]['AddT'] = current_noise
                    Parent = evo.add_noise(model, Parent, device)

                    Off2 = evo.valOff(Parent, min_n_nodes, max_n_nodes, device, max_n_nodes, nodes_dist)
                    Off = evo.denoise_same_level(Off2 + Parent, model, max_n_nodes, device, dataset_info, preds, True, iter_num=f"{run_idx}-{gen}")
                    Pop = evo.Get_Fitness_multi_dis_MAE_normalize(Off + Pop, dataset_info, device, preds, max_n_nodes, Obj, None,
                                                                  0, ObjValue)
                    Pop = evo.EnvironmentalSelectionSingle(Pop, NPops)

                    current_noise = scheduler.update(Off=Off[0:ParentSize-1], Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj, calHV=False)
                    tracker.update(Pop, Obj, MMAE=np.mean([item['MAE'] for item in Pop], axis=0), AddNoise=current_noise, score=scheduler.score)
                    print(f"  {MOEA_NAME} {use_EDM} | Gen {gen + 1} |  Noise: {current_noise} | MMAE: {tracker.metrics['MMAE'][-1]:.4f}   ")

                # 4. 重点：单次运行结束后，一键绘制演化追踪图 (自适应 1D/2D/3D)
                viz.plot_matching_trace(tracker, Obj, ref_point=ObjValue, suffix=f"N{NPops}_Run{run_idx}", interval=2)
                multi_tracker.add('AvgAddT', tracker.metrics['AddNoise'])  # 添加到统计器
                multi_tracker.add('Score', tracker.metrics['score'])  # 添加到统计器
                multi_tracker.add('MMAE', tracker.metrics['MMAE'])  # 添加到统计器


            # 5. 20 轮跑完后，绘制均值收敛汇总图
            # print(f"  正在计算并保存 {Obj} 的统计结果...")
            mean_mmae, std_mmae = multi_tracker.get_aggregated_stats('MMAE')
            viz.plot_mean_curve_with_std(mean_mmae, std_mmae, "MMAE",
                                         f"Mean MMAE over {numberoforun} Runs on {Obj}",
                                         f"MOP_MMAE_{Obj}_{MOEA_NAME}")

            mean_sc, std_sc = multi_tracker.get_aggregated_stats('Score')
            viz.plot_mean_curve_with_std(mean_sc, std_sc, "Score",
                                         f"Mean score over {numberoforun} Runs on {Obj}",
                                         f"MOP_SCORE_{Obj}_{MOEA_NAME}")

            mean_an, std_an = multi_tracker.get_aggregated_stats('AvgAddT')
            viz.plot_mean_curve_with_std(mean_an, std_an, "AvgAddT",
                                         f"Mean AvgAddT over {numberoforun} Runs on {Obj}",
                                         f"MOP_AN_{Obj}_{MOEA_NAME}")

            summary_csv_path = os.path.join(save_path, f"Aggregate_Statson_{Obj}_{MOEA_NAME}.csv")
            with open(summary_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 更新表头
                writer.writerow(
                    ['Gen', 'Mean_HV', 'Std_HV', 'Mean_AN', 'Std_AN', 'Mean_Score', 'Std_Score'])

                # 确保长度一致 (以防万一)
                length = min(len(mean_mmae), len(mean_an))
                for i in range(length):
                    writer.writerow([
                        i,
                        mean_mmae[i], std_mmae[i],
                        mean_an[i], std_an[i],
                        mean_sc[i], std_sc[i]
                    ])

    # print("\n>>> 所有目标匹配实验已完成。")



if __name__ == "__main__":
    main()
