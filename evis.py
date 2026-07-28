import itertools
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
try:
    from pymoo.indicators.hv import Hypervolume
except ImportError:
    from demo_runtime.hypervolume import Hypervolume

import numpy as np


class MultiRunTracker:
    """专门用于跨多次运行的统计器 (支持不同长度序列的自动补齐)"""

    def __init__(self):
        self.data = {}

    def add(self, metric_name, values_list):
        if metric_name not in self.data:
            self.data[metric_name] = []
        self.data[metric_name].append(values_list)

    def get_aggregated_stats(self, metric_name):
        """返回均值和标准差数组，自动处理长短不一的序列"""
        if metric_name not in self.data or len(self.data[metric_name]) == 0:
            return None, None

        raw_lists = self.data[metric_name]

        # 1. 找出所有 run 中的最大代数长度
        max_len = max(len(lst) for lst in raw_lists)
        if max_len == 0:
            return np.array([]), np.array([])

        # 2. 对较短的列表进行 Padding (用最后一个值向后补齐)
        padded_matrix = []
        for lst in raw_lists:
            # 清洗数据，确保是 float 标量
            import torch
            clean_lst = [x.item() if torch.is_tensor(x) else float(x) for x in lst]

            if len(clean_lst) == 0:
                padded_matrix.append([0.0] * max_len)
            elif len(clean_lst) < max_len:
                # 用最后一个元素补齐缺少的代数
                pad_size = max_len - len(clean_lst)
                padded_matrix.append(clean_lst + [clean_lst[-1]] * pad_size)
            else:
                padded_matrix.append(clean_lst)

        # 3. 转换为标准的 float64 矩阵[N_runs, Max_Gens]
        # 注意：这里使用的是 padded_matrix，不会再报 ragged sequence 警告了！
        matrix = np.array(padded_matrix, dtype=np.float64)

        # 4. 计算纵向均值和标准差
        mean_curve = np.mean(matrix, axis=0)
        std_curve = np.std(matrix, axis=0)

        return mean_curve, std_curve

class EvoTracker:
    def __init__(self):
        self._demo_internal_run = -1
        self.reset()

    def reset(self):
        self._demo_internal_run += 1
        self._demo_generation = 0
        self.metrics = {}  # 存储 FR, SuccessRate, MeanVina, BestVina 等
        self.obj_raw = {}  # 存储每一代所有个体的 vina 原始值
        self.obj_feas = {}  # 存储每一代有效个体的 vina 值

    def update(self, pop, obj_list, **additional_metrics):
        # 1. 更新单值指标 (如 FR, AvgCon)
        for key, value in additional_metrics.items():
            if key not in self.metrics: self.metrics[key] = []
            # 确保存入的是标量 float
            val = value.item() if torch.is_tensor(value) else float(value)
            self.metrics[key].append(val)

        # 2. 更新目标值
        for obj in obj_list:
            if obj not in self.obj_raw:
                self.obj_raw[obj], self.obj_feas[obj] = [], []

            self.obj_raw[obj].append([item[obj].item() for item in pop])
            self.obj_feas[obj].append([
                item[obj].item() if sum(item.get('structure_Con', [])) == 0 else None
                for item in pop
            ])

        # Optional append-only experiment recording.  The hook is inert for
        # normal direct script execution and is enabled by the suite launcher
        # through DEMO_RUN_* environment variables.
        try:
            from demo_runtime.recording import record_tracker_update
            record_tracker_update(self, pop, obj_list, additional_metrics)
        except Exception as exc:
            # Recording must never alter the research algorithm's behavior.
            # Persist the error when possible, but keep the legacy run alive.
            try:
                from demo_runtime.recording import get_env_recorder
                recorder = get_env_recorder()
                if recorder is not None:
                    recorder.event("recording_error", error=repr(exc))
            except Exception:
                pass

    def calculate_docking_stats(self, pop, ref_obj=None):
        """
        计算对接统计信息，并加入与原配体的支配关系对比
        ref_obj: 原配体的目标值字典，例如 {'vina': -9.5, 'qed': -0.6, 'sa': 2.5}
        """
        vina_scores = [p['vina'] for p in pop if p['vina'] < 0 and p['Constraint'] <= 0]
        feasible_count = sum(1 for p in pop if p['Constraint'] <= 0)
        qed_scores = [p['qed'] for p in pop if p['vina'] < 0 and p['Constraint'] <= 0]
        sa_scores = [p['sa'] for p in pop if p['vina'] < 10 and p['Constraint'] <= 0]

        dominate_count = 0
        nondom_count = 0

        # 如果传入了参考配体，则计算支配比例
        if ref_obj is not None and len(pop) > 0:
            r_vals = [ref_obj['ref_vina'], ref_obj['ref_qed'], ref_obj['ref_sa']]

            for p in pop:
                if 'vina' in p and 'qed' in p and 'sa' in p and p['Constraint'] <= 0:
                    c_vals = [p['vina'], p['qed'], p['sa']]

                    # 最小化问题下的支配判断逻辑：
                    # 候选解(c) 支配 参考解(r)：c的所有目标 <= r，且至少有一个目标 < r
                    c_dom_r = all(c <= r for c, r in zip(c_vals, r_vals)) and any(c < r for c, r in zip(c_vals, r_vals))

                    # 参考解(r) 支配 候选解(c)：r的所有目标 <= c，且至少有一个目标 < c
                    r_dom_c = all(r <= c for c, r in zip(c_vals, r_vals)) and any(r < c for c, r in zip(c_vals, r_vals))

                    if c_dom_r:
                        dominate_count += 1  # 全面超越原配体
                    elif not r_dom_c:
                        nondom_count += 1  # 互不支配 (各有千秋)

        pop_size = len(pop) if len(pop) > 0 else 1

        return {
            "mean_vina": np.mean(vina_scores) if vina_scores else 0,
            "best_vina": min(vina_scores) if vina_scores else 0,
            "fr": feasible_count / pop_size,
            "count": feasible_count,
            "mean_qed": np.mean(qed_scores) if qed_scores else 0,
            "best_qed": min(qed_scores) if qed_scores else 0,
            "mean_sa": np.mean(sa_scores) if sa_scores else 0,
            "best_sa": min(sa_scores) if sa_scores else 0,
            # 新增的统计字段
            "dom_rate": dominate_count / pop_size,
            "nondom_rate": nondom_count / pop_size
        }

    def get_history(self, obj_list):
        return [self.obj_raw[obj] for obj in obj_list], [self.obj_feas[obj] for obj in obj_list]


class EvoVisualizer:
    def __init__(self, save_dir, dataset_info, plot_3d_molecule_func=None, train_data=None, plot_func=None):
        self.save_dir = save_dir
        self.dataset_info = dataset_info
        self.train_data = train_data
        self.plot_func = plot_func
        self.plot_3d_molecule_func = plot_3d_molecule_func
        self.conversion = {'alpha': 1.0, 'gap': 1000.0, 'homo': 1000.0,
                           'lumo': 1000.0, 'mu': 1.0, 'Cv': 1.0}
        if not os.path.exists(save_dir): os.makedirs(save_dir)

    def _safe_to_numpy(self, data):
        """
        【关键修复】强制将各种嵌套的 List/Tensor 转换为扁平的 float64 numpy 数组
        解决 ValueError: setting an array element with a sequence
        """
        if data is None: return None

        # 递归展开函数
        def flatten(item):
            if isinstance(item, (list, tuple)):
                return [sub for x in item for sub in flatten(x)]
            if torch.is_tensor(item):
                return [item.detach().cpu().item()]
            if isinstance(item, np.ndarray):
                return item.flatten().tolist()
            return [float(item)]

        try:
            # 先展平为纯 Python list
            flat_list = flatten(data)
            return np.array(flat_list, dtype=np.float64)
        except Exception as e:
            print(f"Warning: Data conversion failed: {e}")
            return np.array([])

    def _add_background(self, ax, samples, ref_point, obj_names, is_3d):
        if samples is not None:
            # 使用修复后的函数清洗数据
            s_x = self._safe_to_numpy(samples[0])
            s_y = self._safe_to_numpy(samples[1])
            s_z = self._safe_to_numpy(samples[2]) if is_3d and len(samples) > 2 else None

            # 对齐长度
            min_len = len(s_x)
            if len(s_y) < min_len: min_len = len(s_y)
            if is_3d and s_z is not None and len(s_z) < min_len: min_len = len(s_z)

            if min_len > 0:
                if is_3d:
                    ax.scatter(s_x[:min_len], s_y[:min_len], s_z[:min_len],
                               color='grey', label='samples', alpha=0.6)
                else:
                    ax.scatter(s_x[:min_len], s_y[:min_len],
                               color='grey', label='samples', alpha=0.6)

        if ref_point is not None:
            # 同样清洗参考点
            vals = []
            for name in obj_names:
                val = ref_point.get(name)
                # 处理 Tensor 或 float
                if torch.is_tensor(val): val = val.detach().cpu().item()
                vals.append(val)

            if is_3d:
                ax.scatter(vals[0], vals[1], vals[2], color='red', label='Ref', alpha=1.0, marker='*', s=150,
                           edgecolors='black')
            else:
                ax.scatter(vals[0], vals[1], color='red', label='Ref', alpha=1.0, marker='*', s=150, edgecolors='black')

    def plot_matching_trace(self, tracker, obj_names, ref_point, suffix="", interval=2):
        """
        维度自适应的目标匹配追踪图 (替换原 savedualobj_refpoint)
        """
        n_dim = len(obj_names)
        raw_hists, _ = tracker.get_history(obj_names)
        num_gens = len(raw_hists[0])
        colors = plt.cm.jet(np.linspace(0, 1, num_gens // interval + 1))

        # 统一提取参考点数值
        ref_vals = [ref_point[name].item() if torch.is_tensor(ref_point[name]) else ref_point[name] for name in
                    obj_names]

        # --- 1D 情况：收敛曲线 ---
        if n_dim == 1:
            plt.figure(figsize=(8, 6))
            for i in range(num_gens):
                if i % interval == 0 or i == num_gens - 1:
                    color_idx = i // interval
                    plt.scatter([i] * len(raw_hists[0][i]), raw_hists[0][i], color=colors[color_idx], alpha=0.5, s=10)
            plt.axhline(y=ref_vals[0], color='grey', linestyle='--', label='Target')
            plt.title(f"Matching Trace: {obj_names[0]}")
            plt.xlabel("Generation");
            plt.ylabel(obj_names[0])

        # --- 2D 情况：平面散点 ---
        elif n_dim == 2:
            plt.figure(figsize=(8, 6))
            for i in range(num_gens):
                if i % interval == 0 or i == num_gens - 1:
                    color_idx = i // interval
                    plt.scatter(raw_hists[0][i], raw_hists[1][i], label=f"gen{i}", alpha=0.7, color=colors[color_idx])
            plt.scatter(ref_vals[0], ref_vals[1], color='black', label='Target', marker='*', s=150, zorder=5)
            plt.title(f"Matching Trace: {obj_names[0]}-{obj_names[1]}")
            plt.xlabel(obj_names[0]);
            plt.ylabel(obj_names[1])

        # --- 3D 情况：空间散点 ---
        elif n_dim == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for i in range(num_gens):
                if i % interval == 0 or i == num_gens - 1:
                    color_idx = i // interval
                    ax.scatter(raw_hists[0][i], raw_hists[1][i], raw_hists[2][i], label=f"gen{i}", alpha=0.6,
                               color=colors[color_idx])
            ax.scatter(ref_vals[0], ref_vals[1], ref_vals[2], color='black', label='Target', marker='*', s=200,
                       zorder=10)
            ax.set_title(f"Matching Trace: 3D")
            ax.set_xlabel(obj_names[0]);
            ax.set_ylabel(obj_names[1]);
            ax.set_zlabel(obj_names[2])

        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True, alpha=0.3)
        save_name = f"Trace_{'_'.join(obj_names)}_{suffix}.png"
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_convergence_summary(self, mean_mmae, std_mmae, obj_names, suffix):
        """汇总所有维度的 MAE 收敛曲线 (带标准差阴影)"""
        plt.figure(figsize=(8, 5))
        gens = np.arange(mean_mmae.shape[0])
        for i, name in enumerate(obj_names):
            plt.plot(gens, mean_mmae[:, i], label=f'MAE: {name}', marker='o', markersize=3)
            plt.fill_between(gens, mean_mmae[:, i] - std_mmae[:, i], mean_mmae[:, i] + std_mmae[:, i], alpha=0.15)
        plt.title(f"Overall MAE Convergence ({suffix})")
        plt.xlabel("Generation");
        plt.ylabel("MAE");
        plt.legend();
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"Summary_Conv_{'_'.join(obj_names)}_{suffix}.png"), dpi=300)
        plt.close()

    def plot_convergence(self, mean_mmae, std_mmae, obj_names, title_suffix):
        """
        为一个实验配置绘制所有目标的收敛曲线
        mean_mmae: [gen, n_objs]
        """
        plt.figure(figsize=(8, 5))
        gens = np.arange(mean_mmae.shape[0])
        for i, name in enumerate(obj_names):
            plt.plot(gens, mean_mmae[:, i], label=f'MAE: {name}', marker='o', markersize=3)
            plt.fill_between(gens, mean_mmae[:, i] - std_mmae[:, i],
                             mean_mmae[:, i] + std_mmae[:, i], alpha=0.15)

        plt.title(f"Convergence Trends ({title_suffix})")
        plt.xlabel("Generation")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"Conv_{'_'.join(obj_names)}_{title_suffix}.png"), dpi=300)
        plt.close()

    def get_norm_bounds(self, obj_names):
        """准确提取训练集的 min/max"""
        min_vals, max_vals = [], []
        for name in obj_names:
            raw_train = self.train_data[name]
            # 强制转换为 float 数组
            if torch.is_tensor(raw_train):
                train_np = raw_train.detach().cpu().numpy().astype(float)
            else:
                train_np = np.array([float(x) for x in raw_train])

            scaled_train = train_np * self.conversion.get(name, 1.0)
            min_vals.append(float(np.min(scaled_train)))
            max_vals.append(float(np.max(scaled_train)))
        return min_vals, max_vals

    def calculate_single_gen_hv_docking(self, pop, obj_names, ref_info):
        """
        【新增】专为 Docking 任务设计的 HV 计算函数。
        利用原配体(ref_info)和物理边界进行绝对归一化。
        """
        import torch
        import numpy as np
        n_objs = len(obj_names)

        # 1. 提取完全合规的可行解
        valid_items = []
        for item in pop:
            # 化学约束
            chem_con = max(0, item.get('Constraint', 0))
            # 结构约束 (处理 list 或 标量)
            s_con = item.get('structure_Con', 0)
            struct_con_sum = sum(s_con) if isinstance(s_con, list) else s_con

            # 必须化学合法、结构合法，且 Vina 分数有效 (< 0)
            if chem_con + max(0, struct_con_sum) <= 0 and item.get('vina', 0) < 0:
                valid_items.append(item)

        if not valid_items:
            return 0.0

        # 2. 获取基准值 (防除零保护)
        ref_vina = ref_info.get('ref_vina', -1.0) if ref_info else -1.0
        if ref_vina >= 0: ref_vina = -1e-5  # Vina通常为负，兜底保护

        # 3. 定制化物理归一化
        # 目标逻辑 (最小化)：映射后 1.0 为最差(无贡献)，越小越好(负数更优)
        gen_pts = []
        for name in obj_names:
            # 提取原始值
            raw_vals = np.array([
                it[name].item() if torch.is_tensor(it[name]) else it[name]
                for it in valid_items
            ])

            norm_vals = np.zeros_like(raw_vals, dtype=np.float64)

            if name == 'vina':
                # Vina: 0分最差(映射为1.0)，原配体ref_vina为及格线(映射为0.0)
                # 例如: ref=-10, 生成=-12 -> 1.0 - (-12/-10) = -0.2 (体积超大，完美奖励)
                norm_vals = 1.0 - (raw_vals / ref_vina)

            elif name == 'qed':
                # QED (已取负): 0最差(映射为1.0)，-1最好(映射为0.0)
                norm_vals = raw_vals + 1.0

            elif name == 'sa':
                # SA: 10最差(映射为1.0)，1最好(映射为0.0)
                norm_vals = (raw_vals - 1.0) / 9.0

            else:
                # 兜底
                norm_vals = raw_vals

            gen_pts.append(norm_vals)

        # 4. 计算 HV
        pts = np.column_stack(gen_pts)

        # 参考点设为[1.0, 1.0, ...]
        # 任何大于 1.0 的维度(即比最差情况还差)将不会贡献任何 Hypervolume
        ref_point = np.array([1.0] * n_objs)
        metric = Hypervolume(ref_point=ref_point)

        try:
            return metric.do(pts)
        except Exception as e:
            # print(f"HV calc error: {e}")
            return 0.0

    def calculate_single_gen_hv(self, pop, obj_names, minimize=True):
        """
        【新增】只计算当前这一代种群的 HV，不涉及历史记录
        """
        n_objs = len(obj_names)
        min
        min_b, max_b = self.get_norm_bounds(obj_names)

        # 1. 提取有效解
        valid_items = [
            item for item in pop
            if sum(item.get('structure_Con', [])) == 0  # 结构有效
               and item.get('Constraint', 0) == 0 # 属性有效
        ]

        if not valid_items:
            return 0.0

        # 2. 归一化处理
        gen_pts = []
        for o_idx, name in enumerate(obj_names):
            # 提取原始值
            raw = np.array([it[name].item() if torch.is_tensor(it[name]) else it[name] for it in valid_items])
            # 单位换算
            # scaled = raw * self.conversion.get(name, 1.0)
            # 归一化
            norm = (raw - min_b[o_idx]) / (max_b[o_idx] - min_b[o_idx] + 1e-8)
            # 最小化/最大化处理
            if not minimize: norm = 1.0 - norm
            gen_pts.append(norm)

        # 3. 计算 HV
        pts = np.column_stack(gen_pts)
        metric = Hypervolume(ref_point=np.array([1] * n_objs))
        try:
            return metric.do(pts)
        except:
            return 0.0

    def calculate_hv_list(self, tracker, obj_names, minimize=True):
        """计算单次运行的 HV 列表"""
        _, feas_hist = tracker.get_history(obj_names)
        n_gen, n_objs = len(feas_hist[0]), len(obj_names)
        min_b, max_b = self.get_norm_bounds(obj_names)

        metric = Hypervolume(ref_point=np.array([1] * n_objs))
        hv_history = []

        for g in range(n_gen):
            valid_idx = [i for i in range(len(feas_hist[0][g]))
                         if all(feas_hist[o][g][i] is not None for o in range(n_objs))]

            if not valid_idx:
                hv_history.append(0.0);
                continue

            gen_pts = []
            for o_idx in range(n_objs):
                name = obj_names[o_idx]
                raw = np.array([feas_hist[o_idx][g][i] for i in valid_idx])
                scaled = raw * self.conversion.get(name, 1.0)
                norm = (raw - min_b[o_idx]) / (max_b[o_idx] - min_b[o_idx] + 1e-8)
                if not minimize: norm = 1.0 - norm
                gen_pts.append(norm)

            pts = np.column_stack(gen_pts)
            try:
                hv_history.append(metric.do(pts))
            except:
                hv_history.append(0.0)
        return hv_history

    def plot_mean_curve_with_std(self, mean_vals, std_vals, metric_name, title, suffix):
        """
        【新增】绘制平均曲线带方差阴影
        """
        plt.figure(figsize=(8, 5))
        gens = np.arange(len(mean_vals))

        plt.plot(gens, mean_vals, label=f'Mean {metric_name}', marker='o', markersize=4, color='b')
        plt.fill_between(gens, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color='b', label='Std Dev')

        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"MeanCurve_{metric_name}_{suffix}.png"), dpi=300)
        plt.close()

    def plot_mean_hv(self, all_hv_runs, obj_names, npops):
        """对应你脚本最后绘制 mean_HV 的功能"""
        all_hv_runs = np.array(all_hv_runs)  # [runs, gens]
        mean_hv = np.mean(all_hv_runs, axis=0)

        plt.figure(figsize=(8, 5))
        plt.plot(mean_hv, label='Mean HV', marker='^', color='dodgerblue')
        plt.title(f'Mean HV | PopSize:{npops} | {obj_names[0]}-{obj_names[1]}')
        plt.xlabel('Generation');
        plt.ylabel('HV Value');
        plt.legend();
        plt.grid(True, alpha=0.3)

        save_name = f"{obj_names[0]}{obj_names[1]}_MeanHV_N{npops}.png"
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
        return mean_hv

    def _flatten_nested_list(self, data):
        """
        核心修正：将 [[f1, f2...], [f3...]] 彻底平铺为 1D Numpy 浮点数组
        """
        if torch.is_tensor(data):
            return data.detach().cpu().numpy().flatten()

        try:
            # 使用 itertools 高效平铺嵌套列表
            flattened = list(itertools.chain.from_iterable(data))
            return np.array(flattened, dtype=np.float64)
        except TypeError:
            # 如果本身不是嵌套列表，直接转换
            return np.array(data, dtype=np.float64).flatten()


    def save_stats_csv(self, name, data, obj_names, npops):
        path = os.path.join(self.save_dir, f"{name}_{obj_names}_N{npops}.csv")
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(data)

    def plot_hv_curve(self, tracker, obj_names, ref_point=[1, 1], minimize=False,
                      use_train_bounds=True, manual_min=None, manual_max=None):
        """
        计算并绘制 HV。
        minimize: 你的目标是最小化还是最大化。
                  如果是最大化（如 alpha），设为 False。
        ref_point: 归一化后的参考点。在最小化逻辑下，参考点应设为 [1.1, 1.1]（比最差还差一点）。
        """
        # 1. 确定归一化边界 (你的原始逻辑)
        if use_train_bounds:
            if self.train_data is None:
                raise ValueError("未传入 train_data")
            min_bounds = []
            max_bounds = []
            for name in obj_names:
                # 提取训练集 -> 换算单位 -> 取极值
                train_vals = self.train_data[name] * self.conversion.get(name, 1.0)
                min_bounds.append(train_vals.min().item())
                max_bounds.append(train_vals.max().item())
        else:
            min_bounds, max_bounds = manual_min, manual_max

        # 2. 提取数据
        _, feas_history = tracker.get_history(obj_names)
        n_gen = len(feas_history[0])
        n_objs = len(obj_names)

        # 准备 HV 计算器
        # 注意：pymoo 的 HV 计算是基于最小化假设的
        hv_calculator = Hypervolume(ref_point=np.array(ref_point))
        hv_history = []

        for g in range(n_gen):
            # 对齐有效解索引
            valid_indices = [
                i for i in range(len(feas_history[0][g]))
                if all(feas_history[obj_idx][g][i] is not None for obj_idx in range(n_objs))
            ]

            if not valid_indices:
                hv_history.append(0.0)
                continue

            gen_pts = []
            for obj_idx in range(n_objs):
                name = obj_names[obj_idx]
                raw_vals = np.array([feas_history[obj_idx][g][i] for i in valid_indices])
                # 换算单位
                scaled_vals = raw_vals

                # 归一化：(val - min) / (max - min)
                low, high = min_bounds[obj_idx], max_bounds[obj_idx]
                norm_vals = (scaled_vals - low) / (high - low if high != low else 1.0)

                # --- 核心方向逻辑 ---
                # 如果是最大化问题（如 alpha），归一化后 1 是最好，0 是最差。
                # 为了适配最小化 HV 工具，我们将其翻转：1.0 - norm_vals
                # 翻转后：0 是最好，1 是最差。这才能配合 ref_point=1.1
                if not minimize:
                    norm_vals = 1.0 - norm_vals

                gen_pts.append(norm_vals)

            # 组合为点集 (N, 2)
            pts_matrix = np.column_stack(gen_pts)

            # 计算 HV
            try:
                # 只有被 ref_point 支配的点才有体积。
                # 如果所有点都比 ref_point (1.1) 还差，结果就是 0
                hv_val = hv_calculator.do(pts_matrix)
                hv_history.append(hv_val)
            except Exception as e:
                hv_history.append(0.0)

        # 3. 绘图与保存 (逻辑同前)
        self._save_hv_plot(hv_history, obj_names)
        return hv_history

    def _save_hv_plot(self, hv_history, obj_names):
        plt.figure(figsize=(8, 5))
        plt.plot(hv_history, marker='s', color='blue')
        plt.title(f"HV Curve: {'-'.join(obj_names)}")
        plt.savefig(os.path.join(self.save_dir, f"HV_{'_'.join(obj_names)}.png"))
        plt.close()

    def plot_evolution(self, tracker, obj_names, interval=1, samples=None, ref_point=None,
                       others="", angles=[(30, 45)], only_last_gen=False, show_shadows=True):
        """
        绘制演化图
        - 增强了 3D 空间感（描边、底部投影阴影、透明面板）
        - 增加了 only_last_gen 参数控制是否只画最后一代
        """
        is_3d = (len(obj_names) == 3)
        raw_hist, _ = tracker.get_history(obj_names)
        num_gens = len(raw_hist[0])

        # 1. 确定要绘制的代数 (解决只画最后一代的需求)
        if only_last_gen:
            plot_gens = [num_gens - 1]
        else:
            # 去重并排序，保证按顺序画
            plot_gens = sorted(list(set([i for i in range(num_gens) if i % interval == 0 or i == num_gens - 1])))

        colors = plt.cm.jet(np.linspace(0, 1, num_gens // interval + 1))

        for elev, azim in (angles if is_3d else [(None, None)]):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

            # 绘制背景/参考点
            if not is_3d:
                self._add_background(ax, samples, ref_point, obj_names, is_3d)
            elif ref_point is not None:
                self._add_background(ax, None, ref_point, obj_names, is_3d)

            # 用于收集 3D 坐标，以便稍后在底部画投影阴影
            all_x, all_y, all_z, all_c = [], [], [], []

            # 2. 绘制种群
            for i in plot_gens:
                x_data = self._safe_to_numpy(raw_hist[0][i])
                y_data = self._safe_to_numpy(raw_hist[1][i])

                # 计算颜色：如果只画最后一代，用红色高亮，否则按进度取渐变色
                color_idx = i // interval
                c = colors[color_idx] if not only_last_gen else 'crimson'

                if is_3d:
                    z_data = self._safe_to_numpy(raw_hist[2][i])
                    # 空间感增强1：增加 edgecolors='w', linewidths=0.5 分离重叠点; depthshade=True 开启远近景深
                    ax.scatter(x_data, y_data, z_data, label=f"gen{i}",
                               alpha=0.8, color=c, edgecolors='w', linewidths=0.5, depthshade=True, s=45)

                    if show_shadows:
                        all_x.extend(x_data)
                        all_y.extend(y_data)
                        all_z.extend(z_data)
                        all_c.extend([c] * len(x_data))
                else:
                    ax.scatter(x_data, y_data, label=f"gen{i}",
                               alpha=0.8, color=c, edgecolors='w', linewidths=0.5, s=45)

            # 空间感增强2：绘制底部投影(阴影)
            if is_3d and show_shadows and len(all_z) > 0:
                z_min = min(all_z)
                # 把 Z 坐标全部压到最小值，画半透明的影子
                ax.scatter(all_x, all_y, zs=z_min, zdir='z', c=all_c, alpha=0.15, s=20)

            # 3. 装饰
            ax.set_title(f"Optimization: {'-'.join(obj_names)}")
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])

            if is_3d:
                ax.set_zlabel(obj_names[2])
                ax.view_init(elev=elev, azim=azim)

                # 空间感增强3：将背景墙变为完全透明，仅保留虚线网格
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.grid(True, linestyle='--', alpha=0.6)
            else:
                plt.grid(True, linestyle='--', alpha=0.4)

            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

            # 4. 保存
            angle_str = f"_el{elev}_az{azim}" if is_3d else ""
            last_gen_str = "_LastGenOnly" if only_last_gen else ""

            filename = f"Evo_{'_'.join(obj_names)}_{others}{last_gen_str}{angle_str}.png"
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()

    def plot_evolution_empty(self, tracker, obj_names, interval=1, samples=None, ref_point=None,
                       others="", angles=[(30, 45)], only_last_gen=False, show_shadows=True):
        """
        绘制演化图
        - 根据总约束(化学+结构)决定实心/空心：合法=实心，非法=空心。
        - 增强了 3D 空间感（描边、底部投影阴影、透明面板）
        - 增加了 only_last_gen 参数控制是否只画最后一代
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import torch

        is_3d = (len(obj_names) == 3)
        raw_hist, _ = tracker.get_history(obj_names)
        num_gens = len(raw_hist[0])

        # 尝试从 tracker 提取约束历史
        # 假设 tracker_metrics 里面存了 'Constraint' 和 'structure_Con' 的历史
        # 如果没有存，我们默认为全部合法 (实心)
        chem_con_hist = tracker.metrics.get('Constraint', None)
        struct_con_hist = tracker.metrics.get('structure_Con', None)

        # 1. 确定要绘制的代数 (解决只画最后一代的需求)
        if only_last_gen:
            plot_gens = [num_gens - 1]
        else:
            # 去重并排序，保证按顺序画
            plot_gens = sorted(list(set([i for i in range(num_gens) if i % interval == 0 or i == num_gens - 1])))

        colors = plt.cm.jet(np.linspace(0, 1, num_gens // interval + 1))

        for elev, azim in (angles if is_3d else [(None, None)]):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

            # 绘制背景/参考点
            if not is_3d:
                self._add_background(ax, samples, ref_point, obj_names, is_3d)
            elif ref_point is not None:
                self._add_background(ax, None, ref_point, obj_names, is_3d)

            # 用于收集 3D 坐标，以便稍后在底部画投影阴影
            all_x, all_y, all_z, all_c = [], [], [], []

            # 用于标记是否已经在图例中添加了实心/空心的说明，避免重复
            added_feasible_legend = False
            added_infeasible_legend = False

            # 2. 绘制种群
            for i in plot_gens:
                x_data = self._safe_to_numpy(raw_hist[0][i])
                y_data = self._safe_to_numpy(raw_hist[1][i])
                z_data = self._safe_to_numpy(raw_hist[2][i]) if is_3d else None

                n_points = len(x_data)

                # ================= 计算每个点的合法性 =================
                is_feasible = np.ones(n_points, dtype=bool)  # 默认全是实心合法

                if chem_con_hist is not None and i < len(chem_con_hist):
                    c_con = self._safe_to_numpy(chem_con_hist[i])
                    # 确保是标量数组
                    if c_con is not None and len(c_con) == n_points:
                        c_con = np.maximum(0, c_con)  # 化学约束 <= 0 才合法
                        is_feasible = is_feasible & (c_con <= 0)

                if struct_con_hist is not None and i < len(struct_con_hist):
                    s_con_raw = struct_con_hist[i]
                    s_con_vals = []
                    # 处理可能出现的 list of lists
                    for s in s_con_raw:
                        s_val = sum(s) if isinstance(s, (list, np.ndarray)) else s
                        s_con_vals.append(max(0, s_val))
                    s_con_vals = np.array(s_con_vals)
                    if len(s_con_vals) == n_points:
                        is_feasible = is_feasible & (s_con_vals <= 0)

                # 区分合法点和非法点的索引
                idx_feas = np.where(is_feasible)[0]
                idx_infeas = np.where(~is_feasible)[0]

                # 计算颜色
                color_idx = i // interval
                c = colors[color_idx] if not only_last_gen else 'crimson'

                # ================= 分批绘制 (合法=实心, 非法=空心) =================

                # 绘制合法的点 (Feasible -> Solid)
                if len(idx_feas) > 0:
                    label_str = f"gen{i}" if not added_feasible_legend else "_nolegend_"
                    if is_3d:
                        ax.scatter(x_data[idx_feas], y_data[idx_feas], z_data[idx_feas],
                                   label=label_str, alpha=0.8, color=c, edgecolors='w',
                                   linewidths=0.5, depthshade=True, s=45)
                    else:
                        ax.scatter(x_data[idx_feas], y_data[idx_feas],
                                   label=label_str, alpha=0.8, color=c, edgecolors='w',
                                   linewidths=0.5, s=45)
                    added_feasible_legend = True

                # 绘制非法的点 (Infeasible -> Hollow)
                if len(idx_infeas) > 0:
                    # 如果这代全是空心，为了让图例显示 gen 名字，用第一个空心代显示 label
                    label_str = f"gen{i} (Infeas)" if len(idx_feas) == 0 else "_nolegend_"

                    if is_3d:
                        ax.scatter(x_data[idx_infeas], y_data[idx_infeas], z_data[idx_infeas],
                                   label=label_str, alpha=0.6, facecolors='none', edgecolors=c,
                                   linewidths=1.5, depthshade=True, s=45)
                    else:
                        ax.scatter(x_data[idx_infeas], y_data[idx_infeas],
                                   label=label_str, alpha=0.6, facecolors='none', edgecolors=c,
                                   linewidths=1.5, s=45)
                    added_infeasible_legend = True

                # 收集用于画阴影的 3D 坐标
                if is_3d and show_shadows:
                    all_x.extend(x_data)
                    all_y.extend(y_data)
                    all_z.extend(z_data)
                    all_c.extend([c] * n_points)

            # 空间感增强2：绘制底部投影(阴影)
            if is_3d and show_shadows and len(all_z) > 0:
                z_min = min(all_z)
                # 把 Z 坐标全部压到最小值，画半透明的影子
                ax.scatter(all_x, all_y, zs=z_min, zdir='z', c=all_c, alpha=0.15, s=20)

            # 3. 装饰
            ax.set_title(f"Optimization: {'-'.join(obj_names)}")
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])

            if is_3d:
                ax.set_zlabel(obj_names[2])
                ax.view_init(elev=elev, azim=azim)

                # 空间感增强3：将背景墙变为完全透明，仅保留虚线网格
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.grid(True, linestyle='--', alpha=0.6)
            else:
                plt.grid(True, linestyle='--', alpha=0.4)

            # 增加一个假的图例条目，用来告诉读者实心和空心的含义
            from matplotlib.lines import Line2D
            handles, labels = ax.get_legend_handles_labels()
            handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Feasible'))
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='gray',
                                  markeredgewidth=1.5, markersize=8, label='Infeasible'))
            labels.extend(['Feasible', 'Infeasible'])

            ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1.05, 1))

            # 4. 保存
            angle_str = f"_el{elev}_az{azim}" if is_3d else ""
            last_gen_str = "_LastGenOnly" if only_last_gen else ""

            filename = f"Evo_{'_'.join(obj_names)}_{others}{last_gen_str}{angle_str}.png"
            plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()

    import numpy as np
    import math
    import torch
    import os

    # 增加 auto_focus_highlight 形参，默认开启
    def save_molecule_3d(self, molecule, prefix_type, index, auto_focus_highlight=True):
        """保存单个分子的 3D 渲染图 (save3D 功能)"""
        x, long = molecule['x'], molecule['long']
        atom_type = molecule.get('atom_type', math.nan)
        if isinstance(atom_type, torch.Tensor):
            atom_type = atom_type.detach().cpu()

        # 获取匹配索引
        highlight_idx = molecule.get('matched_indices', None)

        # 截取有效坐标并转换为 cpu
        positions = x[:long].detach().cpu()

        # ==========================================
        # [新增] 自动调整视角的计算逻辑
        # ==========================================
        view_elev, view_azim = None, None

        if auto_focus_highlight:
            pos_np = positions.numpy()
            center_all = np.mean(pos_np, axis=0)

            # 【场景 1】：传入了高亮原子 -> 视角正对高亮部分
            if highlight_idx is not None and len(highlight_idx) > 0:
                highlight_list = list(highlight_idx)
                center_highlight = np.mean(pos_np[highlight_list], axis=0)

                vec = center_highlight - center_all
                norm = np.linalg.norm(vec)

                if norm > 1e-4:
                    vec = vec / norm  # 归一化
                    vx, vy, vz = vec[0], vec[1], vec[2]
                    view_elev = np.degrees(np.arcsin(vz))
                    view_azim = np.degrees(np.arctan2(vy, vx))

            # 【场景 2】：没有高亮原子 -> 使用 PCA 计算分子的最佳展开视角
            else:
                if len(pos_np) > 2:  # 至少需要3个原子才能算平面
                    # 1. 坐标中心化
                    centered_pos = pos_np - center_all
                    # 2. 计算 3x3 协方差矩阵
                    cov = np.cov(centered_pos, rowvar=False)
                    # 3. 特征值分解
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # 4. eigh 返回的特征值默认从小到大排列。
                    # 最小特征值对应的特征向量 (index 0) 就是分子最薄方向的法向量
                    normal_vec = eigenvectors[:, 0]

                    # 统一让法向量朝向上方 (z > 0)，防止有些分子被倒过来画
                    if normal_vec[2] < 0:
                        normal_vec = -normal_vec

                    vx, vy, vz = normal_vec[0], normal_vec[1], normal_vec[2]
                    view_elev = np.degrees(np.arcsin(vz))
                    view_azim = np.degrees(np.arctan2(vy, vx))

        # ==========================================

        # 提取各类评价指标用于命名
        m = {k: molecule.get(k, 'nan') for k in
             ['atm_stable', 'mol_stable', 'validity_rdkit', 'uniqueness_rdkit', 'novelty_rdkit']}
        meta_str = f"_as{m['atm_stable']}_ms{m['mol_stable']}_val{m['validity_rdkit']}_uni{m['uniqueness_rdkit']}_nov{m['novelty_rdkit']}"

        filename = f"{prefix_type}_idx{index}{meta_str}.png"
        save_path = os.path.join(self.save_dir, filename)

        # 传入绘图函数
        self.plot_3d_molecule_func(
            positions=positions,
            atom_type=atom_type[:long].detach().cpu(),
            dataset_info=self.dataset_info,
            save_path=save_path,
            spheres_3d=False,
            highlight_idx=highlight_idx,
            # camera_elev=view_elev,  # <--- [新增] 传入计算好的仰角
            # camera_azim=view_azim  # <--- [新增] 传入计算好的方位角
        )

    def visualize_population(self, pop, obj_list, prefix="finalmol"):
        """可视化整个种群 (visualize_Pops 功能)"""
        for i, mol in enumerate(pop):
            # 自动根据目标名生成文件名
            obj_desc = "_".join([f"{name}{mol[name].item():.2f}" for name in obj_list])
            con_desc = f"Con{mol.get('structure_Con', '100')}"
            full_prefix = f"{prefix}_{obj_desc}_{con_desc}_"
            self.save_molecule_3d(mol, full_prefix, i)

    def plot_metrics_trace(self, tracker, obj_names):
        """绘制训练过程中指标(如 FeasibleRate)的变化曲线"""
        plt.figure(figsize=(8, 5))
        for label, values in tracker.metrics.items():
            plt.plot(values, label=label, marker='o')
        plt.title(f"Metrics: {'-'.join(obj_names)}")
        plt.xlabel("Gen");
        plt.ylabel("Value");
        plt.legend();
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"Metrics_{'_'.join(obj_names)}.png"))
        plt.close()

    def save_run_data_to_csv(self, hv_list, fr_list, ac_list, noise_list, sc_list, uvr_list, uuvr_list, filename):
        """
        【修改】增加 noise_list (AddNoise) 列
        """
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['Generation', 'HV', 'FeasibleRate', 'AvgConstraint', 'AddNoise', 'Score'])

            length = len(hv_list)
            for i in range(length):
                # 安全获取各列数据
                fr = fr_list[i] if i < len(fr_list) else 0
                ac = ac_list[i] if i < len(ac_list) else 0
                an = noise_list[i] if i < len(noise_list) else 0
                writer.writerow([i, hv_list[i], fr, ac, an, sc_list[i], uvr_list[i], uuvr_list[i]])

    def plot_metric_trend(self, data_list, metric_name, color, title, filename):
        """
        【新增】绘制单次运行的指标趋势图 (用于 FR 和 AvgConstraint)
        """
        plt.figure(figsize=(8, 5))
        gens = np.arange(len(data_list))
        plt.plot(gens, data_list, label=metric_name, marker='o', markersize=4, color=color)
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    # evis.py 继续添加...

    # [新增方法] 到 EvoVisualizer 类中
    def save_final_feasible_objs(self, pop, obj_names, filename):
        """
        保存最后一代可行解的目标值到 CSV
        """
        filepath = os.path.join(self.save_dir, filename)

        # 1. 提取可行解
        feasible_inds = []
        for item in pop:
            # 兼容 structure_Con 是列表或标量的情况
            con = item.get('structure_Con', 0)
            if isinstance(con, list):
                con = sum(con)

            # 只有总约束为0才算可行
            if con == 0:
                vals = []
                for obj in obj_names:
                    # 提取张量的值
                    v = item[obj].item() if torch.is_tensor(item[obj]) else item[obj]
                    vals.append(v)
                feasible_inds.append(vals)

        # 2. 保存 CSV
        if feasible_inds:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(obj_names)  # 表头
                writer.writerows(feasible_inds)
            # print(f"    Saved {len(feasible_inds)} feasible solutions to {filename}")
        else:
            print(f"    Warning: No feasible solutions found in final generation for {filename}")

    def save_pf_sdfs(self, pop, atom_decoder, evo_module, prefix):
        """
        【新增】保存最终种群中处于 Pareto 前沿的可行分子为 SDF 文件
        """
        # 在你的 FNDS 中，Pareto 前沿 (Rank 0) 的 fitness 通常 < 1
        # 同时确保分子是可行的 (例如 Vina < 0)
        pf_mols = [p for p in pop if p.get('fitness', 100) < 1.0 and p.get('vina', 0) < 0]

        # 如果没有前沿可行解，退而求其次选 Vina 最好的
        if not pf_mols:
            valid_mols = [p for p in pop if p.get('vina', 0) < 0]
            if valid_mols:
                pf_mols = [min(valid_mols, key=lambda x: x.get('vina', 0))]
            else:
                return  # 完全没有对接成功的分子

        for i, mol in enumerate(pf_mols):
            vina_val = mol.get('vina', 0)
            # 注意：保存时将 QED 翻转回正数，方便人类阅读
            qed_val = -mol.get('qed', 0)
            sa_val = mol.get('sa', 0)

            # 构建文件名 (保存在 self.save_dir 下)
            # 例如: 1IEP_PF_0_vina-9.5_qed0.6_sa2.5
            file_name = f"{prefix}_PF_{i}_vina{vina_val:.2f}_qed{qed_val:.2f}_sa{sa_val:.2f}"
            file_path = os.path.join(self.save_dir, file_name)

            # 调用 evo.py 中的 tensor_to_sdf
            evo_module.tensor_to_sdf(mol, file_path, atom_decoder)


# =========================================================
# [新增类] 用于读取不同算法结果并画图对比
# =========================================================
class ResultComparator:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir): os.makedirs(save_dir)

    def load_all_runs_data(self, algo_path, obj_names):
        """
        读取某个算法文件夹下所有 run_x 里面的 Final_Feasible.csv
        返回: numpy array [Total_Points, N_Objs]
        """
        all_points = []
        # 遍历该算法文件夹下的 run_1, run_2 ...
        if not os.path.exists(algo_path):
            print(f"Path not found: {algo_path}")
            return np.array([])

        for root, dirs, files in os.walk(algo_path):
            for file in files:
                if file == "Final_Feasible.csv":
                    full_path = os.path.join(root, file)
                    try:
                        # 读取 CSV (跳过表头)
                        data = np.genfromtxt(full_path, delimiter=',', skip_header=1)
                        if data.ndim == 1 and data.size > 0:
                            data = data.reshape(1, -1)  # 处理只有一行数据的情况
                        if data.size > 0:
                            all_points.append(data)
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")

        if not all_points:
            return np.array([])

        return np.vstack(all_points)

    def plot_comparison(self, algo_paths_dict, obj_names, title_suffix="Comparison"):
        """
        画对比图
        algo_paths_dict: {'DEMO': './EvoResults/DEMO/task8/alpha_gap', 'NSGA-II': '...'}
        """
        n_dim = len(obj_names)
        is_3d = (n_dim == 3)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        markers = ['o', '^', 's', 'D', 'v', '*']

        for i, (algo_name, path) in enumerate(algo_paths_dict.items()):
            # 1. 加载数据
            data = self.load_all_runs_data(path, obj_names)

            if data.size == 0:
                print(f"No feasible data found for {algo_name}")
                continue

            # 2. 绘制散点
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            if is_3d:
                ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                           label=algo_name, c=color, marker=marker, alpha=0.6, s=20)
            else:
                ax.scatter(data[:, 0], data[:, 1],
                           label=algo_name, c=color, marker=marker, alpha=0.6, s=20)

        # 3. 装饰
        ax.set_title(f"Pareto Front Comparison: {'-'.join(obj_names)}")
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        if is_3d: ax.set_zlabel(obj_names[2])

        ax.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.save_dir, f"Compare_{'_'.join(obj_names)}_{title_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {save_path}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def aggregate_and_save_pocket_runs(multi_tracker, metrics_list, protein_name, save_dir):
    """
    处理某个口袋 20 次运行的聚合数据：画均值曲线(带方差)，保存至 Excel。
    """
    print(f"\n>>> Calculating and Plotting Aggregate Stats for {protein_name} (20 Runs)...")
    os.makedirs(save_dir, exist_ok=True)

    excel_data = {}
    max_gens = 0

    # 获取最大代数以对齐
    for m_key in metrics_list:
        mean_vals, std_vals = multi_tracker.get_aggregated_stats(m_key)
        if mean_vals is not None:
            max_gens = max(max_gens, len(mean_vals))

    if max_gens == 0:
        print("    [Warning] No data found in MultiRunTracker.")
        return

    excel_data['Generation'] = np.arange(max_gens)

    # 遍历每个指标，画平均图并存入 Excel 数据字典
    for m_key in metrics_list:
        mean_vals, std_vals = multi_tracker.get_aggregated_stats(m_key)
        if mean_vals is None:
            continue

        # 【核心修正】：将 QED 翻转回人类易读的正数
        plot_mean = mean_vals.copy()
        if 'QED' in m_key:
            plot_mean = -plot_mean

        # 存入字典供 Pandas 使用
        excel_data[f'{m_key}_Mean'] = plot_mean
        excel_data[f'{m_key}_Std'] = std_vals

        # 绘制带方差阴影的平均曲线
        plt.figure(figsize=(8, 5))
        plt.plot(excel_data['Generation'], plot_mean, label=f'Mean {m_key}', color='#1f77b4', linewidth=2)
        plt.fill_between(excel_data['Generation'],
                         plot_mean - std_vals,
                         plot_mean + std_vals,
                         color='#1f77b4', alpha=0.25, label='±1 Std Dev')

        plt.title(f"Average {m_key} Trend over 20 Runs - {protein_name}", fontweight='bold', fontsize=13)
        plt.xlabel("Generation", fontsize=11)
        plt.ylabel(m_key, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()

        # 保存图片
        save_path = os.path.join(save_dir, f"AggAvg_{protein_name}_{m_key}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 导出字典为 Excel (要求安装 pandas 和 openpyxl)
    df = pd.DataFrame(excel_data)
    csv_path = os.path.join(save_dir, f"Aggregate_Stats_{protein_name}_20Runs.csv")

    # index=False 表示不保存最左侧的行号索引
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"    [Done] Saved aggregate plots and CSV to {csv_path}")
