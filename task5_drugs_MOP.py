# Rdkit import should be first, do not move it
import evo
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ModuleNotFoundError:
    pass
from pathlib import Path
import sys
import utils
from configs.datasets_config import geom_with_h
import argparse
from os.path import join
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
import torch
import pickle
from copy import deepcopy
import os
import evis
import csv

# GPU visibility is controlled by scripts/run_suite.sh before torch is imported.

parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true',default=True,
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',default=True,
                    help='Train first stage AutoEncoder model')

# VAE args
parser.add_argument('--latent_nf', type=int, default=2,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity O(steps).
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

parser.add_argument('--n_epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=False,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=4,
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
parser.add_argument('--dataset', type=str, default='geom',
                    help='dataset name')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', default=True,action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')
parser.add_argument('--resume', type=str, default='outputs/drugs_latent2',
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.9999,           # TODO
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=20,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=5000)
parser.add_argument('--normalization_factor', type=float,
                    default=1, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
parser.add_argument('--filter_molecule_size', type=int, default=None,
                    help="Only use molecules below this size.")
parser.add_argument('--sequential', action='store_true',
                    help='Organize data by size to reduce average memory usage.')
args = parser.parse_args()

use_EDM = False
if use_EDM:
    args.resume = 'tfgmodels/EDM_drugs'
dataset_info = geom_with_h

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda'
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
args.context_node_nf = 0

# Create Latent Diffusion Model or Audoencoder
if use_EDM:
    model, nodes_dist, prop_dist = get_model(args, device, dataset_info, None)
else:
    model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, None)

if prop_dist is not None:
    prop_dist.set_normalizer(None)
model = model.to(device)
gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.

def main():
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.resume, 'generative_model.npy'), map_location=device)) #其实是EMA，没改名字而已
    model.eval()
    MOEA_NAME='SPEA2'
    if use_EDM:
        savedir = f"./DEMO/ProteinPocket_MOP_EDM_{MOEA_NAME}/"
    else:
        savedir = f"./DEMO/ProteinPocket_MOP_GeoLDM_{MOEA_NAME}/"

    tracker = evis.EvoTracker()
    NPops, max_iters = 32, 50
    conda_bin_dir = str(Path(sys.executable).parent)
    Obj = ['vina', 'qed', 'sa']

    metrics_to_plot = ['BestVina', 'Vina', 'BestQED', 'QED', 'BestSA', 'SA', 'FeasibleRate', 'DomRate', 'NonDomRate',
                       'HV', 'UVR']

    for pocket_idx in range(10):
        if pocket_idx!=5:
            continue
        # ==========================================
        # 【新增】针对当前蛋白质口袋，初始化一个跨运行的数据追踪器
        # ==========================================
        multi_tracker = evis.MultiRunTracker()

        # 预先获取口袋信息（因为同个口袋的 ref_vina 是固定的，不用循环20次算）
        info = evo.prepare_pocket_info(pocket_idx, savedir, conda_bin_dir)
        print(f"\n==========================================")
        print(f"Starting Pocket: {info['protein_name']}")
        print(f"==========================================")

        for run_idx in range(1, 21):

            # 跳过已处理
            # if any(f.startswith(info['protein_name']) and f.endswith('.sdf') for f in os.listdir(viz.save_dir)):
            #     continue

            penf = 0.7 # 调节倾向：越大越倾向于找拐点，越小越倾向于找最大值
            os.makedirs(info['protein_dir'], exist_ok=True)
            tracker.reset()
            scheduler = evo.AdaptiveNoiseScheduler(
                dataset_info=dataset_info,
                initial_noise=1000,
                min_noise=0,
                max_noise=1000,
                step_size=10,
                drop_factor=0.5,
                penalty_factor=penf
            )

            # 2. 初始种群
            Pop = evo.InitPop(NPops, nodes_dist, args, device, model, dataset_info, prop_dist,
                              False, info['min_n_nodes'], [], info['max_n_nodes'], False)
            Pop = evo.EvalPop_Con(Pop, dataset_info)
            evo.shift_mol_in_Pop(Pop, *info['center'], device)
            evo.evaluate_vina_pop(Pop, 'Init_', info['pocket_name'], conda_bin_dir, info['protein_dir'],
                                  atom_decoder, dataset_info, info['protein_name'], r_g=info['r_g'])
            evo.evaluate_PB_pop(Pop, 'Init_', info['pocket_pdb'], info['protein_dir'], pbname='Constraint')
            viz = evis.EvoVisualizer(save_dir=info['summary_dir'], dataset_info=dataset_info,
                                     train_data={'vina': [info['ref_vina'], 0], 'qed': [0, info['ref_qed']], 'sa': [info['ref_sa'], 10]})
            Pop = evo.Get_Fitness_Pareto(Pop, dataset_info, device, None, info['max_n_nodes'], Obj, None, 0, calCon=False)
            current_noise = scheduler.update(Off=Pop, Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj, ref_info=info)
            stats = tracker.calculate_docking_stats(Pop, ref_obj=info)
            unique_valid_rate = evo.get_population_duplicate_rate(Pop, dataset_info, NPops)
            tracker.update(Pop, Obj, Vina=stats['mean_vina'], BestVina=stats['best_vina'],
                           QED=stats['mean_qed'], BestQED=stats['best_qed'],
                           SA=stats['mean_sa'], BestSA=stats['best_sa'],
                           FeasibleRate=stats['fr'],
                           DomRate=stats['dom_rate'], NonDomRate=stats['nondom_rate'],
                           AddNoise=current_noise, score=scheduler.score, UVR=unique_valid_rate)

            print(f"[{use_EDM}{info['protein_name']:15}] Init | "
                  f"Vina: {stats['best_vina']:6.2f}({stats['mean_vina']:5.2f}) | "
                  f"QED: {stats['best_qed']:5.3f}({stats['mean_qed']:4.3f}) | "
                  f"SA: {stats['best_sa']:5.3f}({stats['mean_sa']:4.3f}) | "
                  f"FR:{stats['fr']:4.2f} | Dom:{stats['dom_rate']:4.2f} | NDom:{stats['nondom_rate']:4.2f} | "
                  f"Noise:{current_noise:5.3f} | Score:{scheduler.score:5.3f} | HV:{tracker.metrics['HV'][-1]:5.3f} | UVR:{unique_valid_rate:5.3f}")
            # 3. 进化循环
            for it in range(max_iters):
                Parent = evo.k_tournament_selection(Pop, 2, int(NPops / 2))
                for p in Parent:
                    p['AddT'] = current_noise
                    p['EvaluatedVina'] = False
                    p['EvaluatedPB'] = False
                Parent = evo.add_noise(model, Parent, device)

                Off = evo.valOff(Parent, info['min_n_nodes'], info['max_n_nodes'], device, info['max_n_nodes'], nodes_dist)
                Off = evo.denoise_same_level(Off + Parent, model, info['max_n_nodes'], device, dataset_info, [], False)

                # 评估 Offspring
                Off = evo.EvalPop_Con(Off, dataset_info)
                oOff = deepcopy(Off)
                evo.shift_mol_in_Pop(Off, *info['center'], device)
                evo.evaluate_vina_pop(Off, f'Iter_{it}_', info['pocket_name'], conda_bin_dir, info['protein_dir'],
                                      atom_decoder, dataset_info, info['protein_name'], oPop=oOff, draw=False,
                                      r_g=info['r_g'])
                evo.evaluate_PB_pop(Off, f'Iter_{it}_', info['pocket_pdb'], info['protein_dir'], pbname='Constraint')
                Pop = evo.Get_Fitness_Pareto(Pop + Off, dataset_info, device, None, info['max_n_nodes'], Obj, None, 0, calCon=False)

                # 选择并更新 Tracker
                current_noise = scheduler.update(Off=Off[0:int(NPops / 2)-1], Pop=Pop, tracker=tracker, viz=viz, obj_names=Obj, ref_info=info)
                Pop = evo.EnvironmentalSelectionCon(Pop, NPops, Obj)
                stats = tracker.calculate_docking_stats(Pop, ref_obj=info)
                unique_valid_rate = evo.get_population_duplicate_rate(Pop, dataset_info, NPops)
                tracker.update(Pop, Obj, Vina=stats['mean_vina'], BestVina=stats['best_vina'],
                               QED=stats['mean_qed'], BestQED=stats['best_qed'],
                               SA=stats['mean_sa'], BestSA=stats['best_sa'],
                               FeasibleRate=stats['fr'],
                               DomRate=stats['dom_rate'], NonDomRate=stats['nondom_rate'],  # <--- 新增
                               AddNoise=current_noise, score=scheduler.score, UVR=unique_valid_rate)

                print(f"[{info['protein_name']:4}] | {use_EDM} Iter {it:3d} | "
                      f"Vina: {stats['best_vina']:6.2f}({stats['mean_vina']:5.2f}) | "
                      f"QED: {stats['best_qed']:5.3f}({stats['mean_qed']:4.3f}) | "
                      f"SA: {stats['best_sa']:5.3f}({stats['mean_sa']:4.3f}) | "
                      f"FR:{stats['fr']:4.2f} | Dom:{stats['dom_rate']:4.2f} | NDom:{stats['nondom_rate']:4.2f} | "
                      f"Noise:{current_noise:5.3f} | Score:{scheduler.score:5.3f} | HV:{tracker.metrics['HV'][-1]:5.3f} | UVR:{unique_valid_rate:5.3f}")
            print(f">>> Saving single run results for {info['protein_name']} (Run {run_idx})...")

            for m_key in metrics_to_plot:
                if m_key in tracker.metrics:
                    # 【核心修复】：加上 f"Run{run_idx}_"，防止 20 次循环画的图互相覆盖！
                    viz.plot_metric_trend(
                        data_list=tracker.metrics[m_key],
                        metric_name=m_key,
                        color="blue",
                        title=f"{m_key} Trend - {info['protein_name']} (Run {run_idx})",
                        filename=f"Run{run_idx}_{info['protein_name']}_{m_key}_Trend.png"
                    )

                    # 【新增】：将当前 Run 的数据喂给 MultiTracker
                    multi_tracker.add(m_key, tracker.metrics[m_key])

            viz.plot_evolution(
                tracker=tracker,
                obj_names=Obj,
                interval=max(1, max_iters // 5),
                samples=None,
                ref_point=info,
                others=f"EvoTrace_Run{run_idx}_{info['protein_name']}",  # 防覆盖
                angles=[(30, 45), (0, 90)] if len(Obj) == 3 else [(None, None)]
            )

            viz.plot_evolution(
                tracker=tracker,
                obj_names=Obj,
                only_last_gen=True,
                samples=None,
                ref_point=info,
                others=f"EvoTrace_Run{run_idx}_{info['protein_name']}",  # 防覆盖
                angles=[(30, 45), (0, 90)] if len(Obj) == 3 else [(None, None)]
            )

            # 保存 SDF 文件时也可以加上 Run{run_idx} 前缀防覆盖
            viz.save_pf_sdfs(
                pop=Pop,
                atom_decoder=atom_decoder,
                evo_module=evo,
                prefix=f"Run{run_idx}_{info['protein_name']}"
            )

            unified_csv_path = os.path.join(savedir, "RUnified_Docking_Results.csv")
            file_exists = os.path.isfile(unified_csv_path)

            with open(unified_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                # 如果文件是新建的，写入表头
                if not file_exists:
                    writer.writerow([
                        'Pocket', 'Ref_Vina', 'Ref_QED', 'Ref_SA',
                        'Gen_Best_Vina', 'Gen_Mean_Vina',
                        'Gen_Best_QED', 'Gen_Mean_QED',
                        'Gen_Best_SA', 'Gen_Mean_SA',
                        'Feasible_Rate', 'Dominate_Rate', 'NonDom_Rate', 'Final_HV', 'UVR'
                    ])

                final_hv = tracker.metrics['HV'][-1] if 'HV' in tracker.metrics else 0

                writer.writerow([
                    info['protein_name'],
                    info['ref_vina'], info['ref_qed'], info['ref_sa'],
                    stats['best_vina'], stats['mean_vina'],
                    -stats['best_qed'], -stats['mean_qed'],  # 翻转回正值
                    stats['best_sa'], stats['mean_sa'],
                    stats['fr'], stats['dom_rate'], stats['nondom_rate'], final_hv, unique_valid_rate
                ])

            print(f"    [Done] Saved all plots, SDFs, and appended to Unified CSV for {info['pocket_name']}.")
        evis.aggregate_and_save_pocket_runs(
                multi_tracker=multi_tracker,
                metrics_list=metrics_to_plot,
                protein_name=info['protein_name'],
                save_dir=info['summary_dir']
            )











if __name__ == "__main__":
    main()
