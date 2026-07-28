# Rdkit import should be first, do not move it
import evo

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import argparse
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model, get_latent_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
from copy import deepcopy
import random
from qm9.visualizer import plot_data3d_highlight_triview
import utils
import numpy as np
import csv
import os
import evis
from rdkit import rdBase
from MOEA import CMOEACD

rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
from demo_runtime.legacy import env_bool
use_EDM = env_bool("DEMO_USE_EDM", default=False)
# GPU visibility is controlled by scripts/run_suite.sh before torch is imported.
parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='qm9_latent2')
parser.add_argument('--xtb', type=bool, default=False)

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
parser.add_argument('--dataset', type=str, default='qm9',
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
args.xtb = None
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def seed_everything(seed=42):
    """固定所有库的随机种子"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证卷积等算子的确定性（会稍微降低速度，但保证复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_even_splits(total_size):
    # 基础分配
    n1 = total_size // 3
    n2 = total_size // 3

    # 强制转偶数 (向下取偶)
    if n1 % 2 != 0: n1 += 1
    if n2 % 2 != 0: n2 -= 1

    # 剩余的都给 FEA (n3)，确保总和不变
    n3 = total_size - n1 - n2
    result = [n1, n2, n3]
    random.shuffle(result)
    return result

def main():
    if args.resume is not None:
        flow_state_dict = torch.load(os.path.join(args.resume, 'generative_model_ema.npy'), map_location=device)
        model.load_state_dict(flow_state_dict)
        model.eval()

        # 进化参数
    NPops = 32
    max_n_nodes = 29
    min_n_nodes = 15
    iters = 50  # 每一轮进化的代数
    total_runs = 20  # 每个目标组合独立运行的次数 (seed 1 to 20)
    num_workers = 9
    USE_HV_TRIGGER = False
    MOEA_NAME = 'HARD_CMOEACD2'
    Ns = int(NPops // 3)


    # 属性组合
    prop = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv']
    # combinations = list(itertools.combinations(prop, 2))

    comb2 = [('alpha', 'gap'), ('gap', 'homo'), ('homo', 'lumo'), ('lumo', 'mu'), ('mu', 'Cv'), ('Cv', 'alpha')]
    comb3 = [ ('alpha', 'homo', 'mu'), ('gap', 'lumo', 'Cv'), ('alpha', 'lumo', 'Cv'), ('gap', 'homo', 'mu')]
    combinations = comb3 + comb2

    print(f"Total tasks: {len(combinations)}")
    print("Task list:", combinations)

    with open('./EvoResults/task0_half/values.pkl', 'rb') as f:
        samplevalues = pickle.load(f)

    if use_EDM:
        base_save_root = f'./DEMO/{MOEA_NAME}_CMOP_EDM_seed0/'
    else:
        base_save_root = f'./DEMO/{MOEA_NAME}_CMOP_GEOLDM_seed0/'

    seed = 10
    # ==========================================
    # 2. 属性组合大循环 (Outer Loop)
    # ==========================================
    for Obj in combinations:
        seed += 1
        # Obj = ('mu','Cv')
        obj_str = "_".join(Obj)
        print(f"\nStarting Task: {Obj}")

        sample1 = samplevalues[prop.index(Obj[0])]
        sample2 = samplevalues[prop.index(Obj[1])]
        current_samples = [sample1, sample2]
        if len(Obj) == 3:
            current_samples.append(samplevalues[prop.index(Obj[2])])

        preds = evo.Get_Pred(Obj, device)
        multi_tracker = evis.MultiRunTracker()

        # ==========================================
        # 3. 独立运行循环 (Run Loop: Seed 1 to 20)
        # ==========================================
        for run_idx in range(1, total_runs + 1):
            tolerance = 0

            current_save_dir = os.path.join(base_save_root, obj_str, f"run_{run_idx}")
            viz = evis.EvoVisualizer(
                save_dir=current_save_dir,
                dataset_info=dataset_info,
                plot_3d_molecule_func=plot_data3d_highlight_triview,
                train_data=dataloaders['train'].dataset.data
            )
            tracker = evis.EvoTracker()

            # --- C. 获取参考片段 (使用当前的 seed) ---
            results = evo.Get_Multipattern(
                args, device, model, dataset_info, prop_dist,
                17, 29, preds, L=[7, 7], seed=seed
            )

            # 解压并保存参考片段
            patts = [res[0] for res in results]
            pattcrops = [res[1] for res in results]

            for i, (p, pc) in enumerate(zip(patts, pattcrops)):
                viz.save_molecule_3d(p, f"patt_ref_{i}", 0)
                viz.save_molecule_3d(pc, f"pattcrop_ref_{i}", 0)

            def augment_pop(population):
                # 计算目标值
                population = evo.Get_Fitness_Pareto(population, dataset_info, device, preds, max_n_nodes, Obj, pattcrops, 0, num_workers=num_workers)
                for p in population:
                    # 构造 objs 向量 (假设全是最小化)
                    p['objs'] = torch.tensor([p[o].item() for o in Obj])
                    # 确保有 structure_Con
                    if 'structure_Con' not in p: p['structure_Con'] = 0
                return population

            # --- D. 初始化种群 ---
            Pop = evo.InitPop(NPops, nodes_dist, args, device, model, dataset_info, prop_dist, False, min_n_nodes,
                              preds, max_n_nodes, True)
            Pop = augment_pop(Pop)
            FA = []
            DA = []
            FEA = Pop  # 初始 FEA 为整个种群
            Offspring = Pop
            all_objs = CMOEACD.get_obj_tensor(Pop)
            zmin = np.min(all_objs, axis=0) - 1e-6

            scheduler = evo.AdaptiveNoiseScheduler(dataset_info=dataset_info, initial_noise=1000, min_noise=0,
                                                   max_noise=1000, step_size=20,
                                                   drop_factor=0.5, use_hv_trigger=USE_HV_TRIGGER)
            current_noise = scheduler.update(Off=Pop, Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj)
            fr_init, init_ac = evo.get_fr_avgcon(Pop)
            unique_valid_rate = evo.get_population_duplicate_rate(Pop, dataset_info, NPops)
            UUVR = evo.get_population_duplicate_rate(Pop, dataset_info, NPops, usecon=False)
            tracker.update(Pop, Obj, FeasibleRate=fr_init, AvgConstraint=init_ac, AddNoise=current_noise, Score=scheduler.score, UVR=unique_valid_rate, UUVR=UUVR)

            # ==========================================
            # 4. 进化迭代循环 (Generation Loop)
            # ==========================================
            for gen in range(iters):
                # 1. 更新 zmin (理想点)
                off_objs = CMOEACD.get_obj_tensor(Offspring) if gen > 0 else CMOEACD.get_obj_tensor(Pop)
                zmin = np.minimum(zmin, np.min(off_objs, axis=0) - 1e-6)

                # 2. 更新三个存档
                FA = CMOEACD.update_FA(FA, Offspring, zmin, Ns)
                DA = CMOEACD.update_DA(DA, Offspring, zmin, Ns)
                FEA = CMOEACD.update_FEA(FEA, Offspring, NPops)

                # 3. 确定各部分父代大小
                # 总父代大小需为 NPops / 2
                total_parent_size = int(NPops / 2)
                n_fa, n_da, n_fea = get_even_splits(total_parent_size)
                # 例如 32 -> 16 -> (4, 4, 8) 或 (6, 4, 6) 取决于整除情况，保证都是偶数

                # 4. 独立选择父代 (Selection)
                # 确保源列表非空
                valid_FA = FA if len(FA) > 0 else FEA
                valid_DA = DA if len(DA) > 0 else FEA

                # 从 FA 选 n_fa 个 (replace=True 允许重复采样，防止存档过小报错)
                idx1 = np.random.choice(len(valid_FA), size=n_fa, replace=True)
                pool1 = [deepcopy(valid_FA[i]) for i in idx1]

                # 从 DA 选 n_da 个
                idx2 = np.random.choice(len(valid_DA), size=n_da, replace=True)
                pool2 = [deepcopy(valid_DA[i]) for i in idx2]

                # 从 FEA 选 n_fea 个
                idx3 = np.random.choice(len(FEA), size=n_fea, replace=True)
                pool3 = [deepcopy(FEA[i]) for i in idx3]

                # 5. 统一加噪 (Batch Processing)
                # 为了利用 GPU 效率，先合并加噪，再拆分
                All_Parents = pool1 + pool2 + pool3
                for i in range(len(All_Parents)):
                    All_Parents[i]['AddT'] = current_noise
                    All_Parents[i]['EvaluatedSC'] = False

                # 加噪
                All_Parents_Noised = evo.add_noise(model, All_Parents, device)

                # 重新拆分回三组
                P1_Noised = All_Parents_Noised[:n_fa]
                P2_Noised = All_Parents_Noised[n_fa: n_fa + n_da]
                P3_Noised = All_Parents_Noised[n_fa + n_da:]

                # 6. 独立交叉 (Separate Crossover)
                # 这里的关键是：FA 的父代只和 FA 的父代交叉，保持探索特性的纯粹

                # pattcrops_noised = []
                # for pc in pattcrops:
                #     pc_tmp = deepcopy(pc)
                #     pc_tmp['AddT'] = current_noise
                #     pattcrops_noised.append(pc_tmp)
                # pattcrops_noised = evo.add_noise(model, pattcrops_noised, device)

                Offpatt = []
                # for pc_noisy in pattcrops_noised:
                #     sub_off = evo.valOff_Patt(pc_noisy, P1_Noised, min_n_nodes, max_n_nodes, device, max_n_nodes,
                #                               nodes_dist)
                #     Offpatt.extend(sub_off)
                #     sub_off2 = evo.valOff_Patt(pc_noisy, P2_Noised, min_n_nodes, max_n_nodes, device, max_n_nodes,
                #                                nodes_dist)
                #     Offpatt.extend(sub_off2)
                #     sub_off3 = evo.valOff_Patt(pc_noisy, P3_Noised, min_n_nodes, max_n_nodes, device, max_n_nodes,
                #                                nodes_dist)
                #     Offpatt.extend(sub_off3)

                Off1 = []
                Off2 = []
                Off3 = []

                if len(P1_Noised) >= 2:
                    Off1 = evo.valOff(P1_Noised, min_n_nodes, max_n_nodes, device, max_n_nodes, nodes_dist)

                if len(P2_Noised) >= 2:
                    Off2 = evo.valOff(P2_Noised, min_n_nodes, max_n_nodes, device, max_n_nodes, nodes_dist)

                if len(P3_Noised) >= 2:
                    Off3 = evo.valOff(P3_Noised, min_n_nodes, max_n_nodes, device, max_n_nodes, nodes_dist)

                # 7. 统一去噪 (Denoising)
                # 按照 DEMO 逻辑，输入是 (Off + Parent)，即去噪修正后的子代和父代
                # 组合所有需要去噪的分子
                lencross = len(Offpatt + Off1 + Off2 + Off3)
                Denoise_Input = Offpatt + Off1 + Off2 + Off3 + P1_Noised + P2_Noised + P3_Noised

                # 批量去噪
                # 结果的总大小应该是 (n_fa*2 + n_da*2 + n_fea*2) = Total_Parent * 2 = NPops
                Offspring = evo.denoise_same_level(Denoise_Input, model, max_n_nodes, device, dataset_info, preds,
                                                   False)

                # 8. 后处理与评估
                Offspring = augment_pop(Offspring)

                # 注意：Environment Selection 下一代的主种群应该是 FEA
                # 在 CMOEA-CD 中，每一代其实是在维护三个存档，FEA 是输出结果
                Pop = FEA

                # 9. 调度器更新
                current_noise = scheduler.update(Off=Offspring[0:lencross-1], Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj)

                fr, avg_con = evo.get_fr_avgcon(Pop)
                unique_valid_rate = evo.get_population_duplicate_rate(Pop, dataset_info, NPops)
                UUVR = evo.get_population_duplicate_rate(Pop, dataset_info, NPops, usecon=False)
                tracker.update(Pop, Obj, FeasibleRate=fr, AvgConstraint=avg_con, AddNoise=current_noise,
                               Score=scheduler.score, UVR=unique_valid_rate, UUVR=UUVR)
                print(f"  {MOEA_NAME} {use_EDM} {len(Offspring)} | Gen {gen+1} | FR: {fr:.2f} | Noise: {current_noise} | HV: {tracker.metrics['HV'][-1]:.4f} | AvgConstraint: {avg_con:.4f} | UVR: {unique_valid_rate:.4f} ")

            # ==========================================
            # 5. 单次运行结果导出 (到 run_x 文件夹)
            # ==========================================
            print(f"    >>> Saving results for Run {run_idx}...")

            viz.save_final_feasible_objs(Pop, Obj, f"Final_Feasible.csv")

            # A. 获取各指标列表
            hv_list = tracker.metrics['HV']
            fr_list = tracker.metrics['FeasibleRate']
            ac_list = tracker.metrics['AvgConstraint']
            an_list = tracker.metrics['AddNoise']
            sc_list = tracker.metrics['Score']
            uvr_list = tracker.metrics['UVR']
            uuvr_list = tracker.metrics['UUVR']

            # B. 保存单次运行 CSV (HV, FR, AC)
            viz.save_run_data_to_csv(hv_list, fr_list, ac_list, an_list, sc_list, uvr_list, uuvr_list,
                                     f"Run{run_idx}_Stats_{Obj}_{MOEA_NAME}.csv")

            # C. 添加到多轮统计器
            multi_tracker.add('HV', hv_list)
            multi_tracker.add('FeasibleRate', fr_list)
            multi_tracker.add('AvgConstraint', ac_list)
            multi_tracker.add('AvgAddT', an_list)
            multi_tracker.add('Score', sc_list)
            multi_tracker.add('UVR', uvr_list)
            multi_tracker.add('UUVR', uuvr_list)

            # 1. 可行率 (Feasible Rate)
            viz.plot_metric_trend(fr_list, "FeasibleRate", "green",
                                  f"Feasible Rate - Run {run_idx} on {Obj}",
                                  f"Run{run_idx}_FeasibleRate on {Obj}_{MOEA_NAME}.png")

            viz.plot_metric_trend(ac_list, "AvgConstraint", "orange",
                                  f"Avg Constraint - Run {run_idx} on {Obj}",
                                  f"Run{run_idx}_AvgConstraint on {Obj}_{MOEA_NAME}.png")

            viz.plot_metric_trend(hv_list, "HV", "blue",
                                  f"Avg Constraint - Run {run_idx} on {Obj}",
                                  f"Run{run_idx}_HV on {Obj}_{MOEA_NAME}.png")

            viz.plot_metric_trend(an_list, "AddNoise", "red",
                                  f"AddNoise - Run {run_idx} on {Obj}",
                                  f"Run{run_idx}_AddNoise on {Obj}_{MOEA_NAME}.png")

            viz.plot_metric_trend(sc_list, "Score", "yellow",
                                  f"Score - Run {run_idx} on {Obj}",
                                  f"Run{run_idx}_Score on {Obj}_{MOEA_NAME}.png")

            viz.plot_metric_trend(uvr_list, "UVR", "grey",
                                  f"Unique and Valid Rate - Run {run_idx} on {Obj}",
                                  f"Run{run_idx}_UVR on {Obj}_{MOEA_NAME}.png")
            viz.plot_metric_trend(uuvr_list, "UUVR", "grey",
                                  f"Unconstrained Unique and Valid Rate - Run {run_idx} on {Obj}",
                                  f"Run{run_idx}_UUVR on {Obj}_{MOEA_NAME}.png")

            viz.plot_evolution(
                tracker, Obj, interval=5,
                samples=current_samples,
                others=f"trace_seed{seed}",
                angles=[(45, 45), (0, 90)] if len(Obj) == 3 else [(None, None)]
            )

            viz.plot_hv_curve(tracker, Obj, use_train_bounds=True, minimize=True)

            viz.visualize_population(Pop, Obj)

            del Pop
            torch.cuda.empty_cache()

        print(f"\n>>> Calculating Aggregate Stats for {Obj}...")

        summary_dir = os.path.join(base_save_root, obj_str, "Summary")
        if not os.path.exists(summary_dir): os.makedirs(summary_dir)
        summary_viz = evis.EvoVisualizer(save_dir=summary_dir, dataset_info=dataset_info)

        # A. 统计 HV
        mean_hv, std_hv = multi_tracker.get_aggregated_stats('HV')
        summary_viz.plot_mean_curve_with_std(mean_hv, std_hv, "Hypervolume",
                                             f"Mean HV over {total_runs} Runs on {Obj}", f"HV_{Obj}_{MOEA_NAME}")

        # B. 统计 FR
        mean_fr, std_fr = multi_tracker.get_aggregated_stats('FeasibleRate')
        summary_viz.plot_mean_curve_with_std(mean_fr, std_fr, "FeasibleRate",
                                             f"Mean FR over {total_runs} Runs on {Obj}", f"FR_{Obj}_{MOEA_NAME}")

        # C. 统计 AvgConstraint
        mean_ac, std_ac = multi_tracker.get_aggregated_stats('AvgConstraint')
        summary_viz.plot_mean_curve_with_std(mean_ac, std_ac, "AvgConstraint",
                                             f"Mean AvgConstraint over {total_runs} Runs on {Obj}",
                                             f"AC_{Obj}_{MOEA_NAME}")

        mean_an, std_an = multi_tracker.get_aggregated_stats('AvgAddT')
        summary_viz.plot_mean_curve_with_std(mean_an, std_an, "AvgAddT",
                                             f"Mean AvgAddT over {total_runs} Runs on {Obj}", f"AN_{Obj}_{MOEA_NAME}")

        mean_sc, std_sc = multi_tracker.get_aggregated_stats('Score')
        summary_viz.plot_mean_curve_with_std(mean_sc, std_sc, "Score",
                                             f"Mean Score over {total_runs} Runs on {Obj}", f"AN_{Obj}_{MOEA_NAME}")

        mean_uvr, std_uvr = multi_tracker.get_aggregated_stats('UVR')
        summary_viz.plot_mean_curve_with_std(mean_uvr, std_uvr, "UVR",
                                             f"Mean UVR over {total_runs} Runs on {Obj}", f"AN_{Obj}_{MOEA_NAME}")

        mean_uuvr, std_uuvr = multi_tracker.get_aggregated_stats('UUVR')
        summary_viz.plot_mean_curve_with_std(mean_uuvr, std_uuvr, "UUVR",
                                             f"Mean UUVR over {total_runs} Runs on {Obj}", f"AN_{Obj}_{MOEA_NAME}")

        # E. 保存汇总 CSV (增加 AvgConstraint 列)
        summary_csv_path = os.path.join(summary_dir, f"Aggregate_Statson_{Obj}_{MOEA_NAME}.csv")
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 更新表头
            writer.writerow(['Gen', 'Mean_HV', 'Std_HV', 'Mean_FR', 'Std_FR', 'Mean_AC', 'Std_AC', 'Mean_AN', 'Std_AN',
                             'Mean_Score', 'Std_Score, MEAN_UVR, STD_UVR', 'MEAN_UUVR, STD_UUVR'])

            # 确保长度一致 (以防万一)
            length = min(len(mean_hv), len(mean_fr), len(mean_ac))
            for i in range(length):
                writer.writerow([
                    i,
                    mean_hv[i], std_hv[i],
                    mean_fr[i], std_fr[i],
                    mean_ac[i], std_ac[i],
                    mean_an[i], std_an[i],
                    mean_sc[i], std_sc[i],
                    mean_uvr[i], std_uvr[i],
                    mean_uuvr[i], std_uuvr[i],
                ])

    print("\nALL EXPERIMENTS COMPLETED")



if __name__ == "__main__":
    main()
