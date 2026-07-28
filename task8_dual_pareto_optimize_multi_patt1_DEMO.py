def main():
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
    import random
    from qm9.visualizer import plot_data3d,save_xyz_file,load_molecule_xyz, plot_data3d_highlight, plot_data3d_highlight_triview
    import utils
    import numpy as np
    import csv
    import os
    import evis
    from rdkit import rdBase
    from copy import deepcopy
    from MOEA import DEMO

    rdBase.DisableLog('rdApp.warning')
    rdBase.DisableLog('rdApp.error')
    use_EDM = True
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
    MOEA_NAME = 'HARD_DEMO2_plot'

    # 属性组合
    prop = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv']
    # combinations = list(itertools.combinations(prop, 2))

    comb2 = [('alpha', 'gap'), ('gap', 'homo'), ('homo', 'lumo'), ('lumo', 'mu'), ('mu', 'Cv'), ('Cv', 'alpha')]
    comb3 = [ ('alpha', 'homo', 'mu'), ('gap', 'lumo', 'Cv'), ('alpha', 'lumo', 'Cv'), ('gap', 'homo', 'mu')]
    comb4 = [('Cv', 'alpha')]
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

            # --- D. 初始化种群 ---
            print(f"\n[Run {run_idx}] Initializing Main and Diversity Populations...")

            # 生成两倍大小的初始池
            Initial_Pool = evo.InitPop(NPops, nodes_dist, args, device, model, dataset_info, prop_dist,
                                       False, min_n_nodes, preds, max_n_nodes, True)

            # 初始评价
            Initial_Pool = evo.Get_Fitness_Pareto(Initial_Pool, dataset_info, device, preds, max_n_nodes, Obj,
                                                  pattcrops, tolerance, num_workers=num_workers)

            # PopMain (主种群): 使用标准的 SPEA2-CDP

            PopMain = DEMO.EnvironmentalSelectionMain_Diverse(deepcopy(Initial_Pool), NPops, Obj, dataset_info)
            Pop_A = deepcopy(PopMain)
            Pop_B = deepcopy(PopMain)
            Pop_C = deepcopy(PopMain)
            # PopDiv = DEMO.EnvironmentalSelectionDiv(Initial_Pool, PopMain, NPops, dataset_info)

            scheduler = evo.AdaptiveNoiseScheduler(dataset_info=dataset_info, initial_noise=1000, min_noise=0,
                                                   max_noise=1000, step_size=20, drop_factor=0.5,
                                                   use_hv_trigger=USE_HV_TRIGGER)
            current_noise = scheduler.update(Off=Pop_A, Pop=Pop_A, tracker=tracker, viz=viz, obj_names=Obj)

            fr_init, init_ac = evo.get_fr_avgcon(Pop_A)
            unique_valid_rate = evo.get_population_duplicate_rate(Pop_C, dataset_info, NPops)
            UUVR = evo.get_population_duplicate_rate(Pop_C, dataset_info, NPops, usecon=False)
            tracker.update(Pop_A, Obj, FeasibleRate=fr_init, AvgConstraint=init_ac, AddNoise=current_noise,
                           Score=scheduler.score, UVR=unique_valid_rate, UUVR=UUVR)

            # ==========================================
            # 4. 进化迭代循环 (Generation Loop)
            # ==========================================
            for gen in range(iters):

                # ==================================
                # 车间 A：底盘自由探索 + 强制组装
                # ==================================
                Parent_A = evo.k_tournament_selection(Pop_A, 2, int(NPops / 2))
                for p in Parent_A: p['AddT'] = current_noise
                Parent_A_Noised = evo.add_noise(model, Parent_A, device)

                # 【新增】：将 A 的父代一分为二
                half_A = len(Parent_A_Noised) // 2
                Parent_A_Internal = Parent_A_Noised[:half_A]
                Parent_A_Forced = Parent_A_Noised[half_A:]

                # 动作 1：A 种群内部基因交流 (维持野生底盘多样性)
                Off_A_Int = []
                if len(Parent_A_Internal) >= 2:
                    Off_A_Int = evo.valOff(Parent_A_Internal, min_n_nodes, max_n_nodes, device, max_n_nodes, nodes_dist)

                # 动作 2：A 与目标片段强制拼接 (组装)
                pattcrops_noised = []
                for pc in pattcrops:
                    pc_tmp = deepcopy(pc)
                    pc_tmp['AddT'] = current_noise
                    pattcrops_noised.append(pc_tmp)
                pattcrops_noised = evo.add_noise(model, pattcrops_noised, device)

                Off_A_Forced = []
                import random
                for p_a in Parent_A_Forced:
                    target_frag = random.choice(pattcrops_noised)
                    sub_off = evo.valOff_Patt(target_frag, [p_a], min_n_nodes, max_n_nodes, device, max_n_nodes,
                                              nodes_dist)
                    Off_A_Forced.extend(sub_off)

                # 汇总 A 产生的子代
                Off_A_Total = Off_A_Int + Off_A_Forced

                # ==================================
                # 车间 B：纯构象打磨 (仅变异，绝不交叉)
                # ==================================
                total_mut_budget = int(NPops / 2)

                # 1. 动态名额分配策略
                if len(Pop_C) == 0:
                    # 前期没有完美品，B 车间满负荷处理半成品
                    n_B = total_mut_budget
                    n_C = 0
                elif len(Pop_B) == 0:
                    # 极端情况：全是完美品，全部分配给 C 进行巅峰微调
                    n_B = 0
                    n_C = total_mut_budget
                else:
                    # 常规情况：B 至少占一半 (向上取整保证 B >= C)
                    n_B = (total_mut_budget // 2) + (total_mut_budget % 2)
                    n_C = total_mut_budget - n_B

                # 2. 独立锦标赛选择 (分离的基因池)
                Parent_B = []
                if n_B > 0:
                    Parent_B.extend(evo.k_tournament_selection(Pop_B, 2, n_B))
                if n_C > 0:
                    Parent_B.extend(evo.k_tournament_selection(Pop_C, 1, n_C))

                # 3. 统一加噪 (构象松弛)
                for p in Parent_B:
                    p['AddT'] = current_noise
                    p['EvaluatedSC'] = False  # 需要重新评估片段

                Parent_B_Noised = evo.add_noise(model, Parent_B, device)

                # ==================================
                # 统一去噪车间 (GPU Batch)
                # ==================================
                # 放入去噪器的包含：A 的内部子代、A 的拼接子代、B 的变异父代
                Candidates_To_Denoise = Off_A_Total + Parent_B_Noised

                Off_Denoised = evo.denoise_same_level(Candidates_To_Denoise, model, max_n_nodes, device, dataset_info,
                                                      preds, True)

                # 统一评价
                Off_Evaluated = DEMO.Get_Fitness_Pareto_Main(Off_Denoised, dataset_info, device, preds, max_n_nodes, Obj,
                                                            pattcrops, tolerance=0)

                # ==================================
                # 货品分发与各车间环境选择
                # ==================================
                # 把 A和B 的祖本，以及新产出的所有商品，丢进中央奖池
                Central_Pool = Pop_A + Pop_B + Off_Evaluated

                # 1. 完美品进入 C 库 (Archive)
                Pop_C = DEMO.EnvironmentalSelection_C_Archive(Central_Pool + Pop_C, NPops, Obj, dataset_info)

                # 2. 半成品进入 B 车间 (严格要求不可行解，向完美拼接努力)
                # 若 Central_Pool 中全都是完美品，函数内部会容错处理
                Pop_B = DEMO.EnvironmentalSelection_B(Central_Pool, NPops, Obj, dataset_info)

                # 3. 游离骨架进入 A 车间 (排斥拼接，向未知的化学空间深处探索)
                Pop_A = DEMO.EnvironmentalSelection_A(Central_Pool, NPops, Obj, dataset_info)

                # ==================================
                # 调度与日志
                # ==================================
                # 调度器监控进度：B 车间的平均约束最能反映组装难度，如果 B 卡住了，就需要调整噪声
                monitor_pop = Pop_B if len(Pop_B) > 0 else Pop_A
                current_noise = scheduler.update(Off=Off_Denoised[0:int(NPops / 2)-1], Pop=Pop_C, tracker=tracker, viz=viz,
                                                 obj_names=Obj)

                fr = len(Pop_C) / NPops if len(Pop_C) > 0 else 0.0
                _, avg_con = evo.get_fr_avgcon(Pop_B)
                unique_valid_rate = evo.get_population_duplicate_rate(Pop_C, dataset_info, NPops)
                UUVR = evo.get_population_duplicate_rate(monitor_pop, dataset_info, NPops, usecon=False)
                tracker.update(monitor_pop, Obj, FeasibleRate=fr, AvgConstraint=avg_con, AddNoise=current_noise,
                               Score=scheduler.score, UVR=unique_valid_rate, UUVR=UUVR)

                print(
                    f"  [Assembly-Line] Gen {gen + 1:03d} {len(Off_Denoised)} {MOEA_NAME}|FR(C): {fr} | AC(B): {avg_con:.2f} | Noise: {current_noise} | HV(C): {tracker.metrics['HV'][-1]:.4f} | UVR: {unique_valid_rate:.4f}")


            # ==========================================
            # 5. 单次运行结果导出 (到 run_x 文件夹)
            # ==========================================
            Pop = Pop_C
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
                tracker, Obj, interval=25,
                samples=current_samples,
                others=f"trace_seed{seed}",
                angles=[(45, 45), (0, 90)] if len(Obj) == 3 else [(None, None)],
                only_last_gen=True
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
