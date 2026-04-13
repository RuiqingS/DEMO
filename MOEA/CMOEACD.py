import evo
import torch
import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist


def get_obj_tensor(pop):
    """辅助函数：提取目标值矩阵 [N, M]"""
    return torch.stack([item['objs'] for item in pop]).cpu().numpy()


def get_specific_cv(pop, ignore_structural=False):
    """
    根据需求提取 CV 值。
    如果 ignore_structural=True，CV 只包含化学约束。
    如果 ignore_structural=False，CV = 化学约束 + 结构约束。
    """
    cv_list = []
    for item in pop:
        # 1. 化学约束 (始终考虑)
        chem = max(0, item.get('Constraint', 0))

        # 2. 结构约束 (可选考虑)
        if ignore_structural:
            struct = 0
        else:
            s_con = item.get('structure_Con', 0)
            if isinstance(s_con, list):
                struct = sum(s_con)
            else:
                struct = s_con
            struct = max(0, struct)

        cv_list.append(chem + struct)
    return np.array(cv_list)


def calc_domination(pop, ignore_structural=False):
    """
    计算支配关系 (CDP 规则)。

    Args:
        pop: 种群
        ignore_structural:
            True  -> FA 模式 (只考虑化学约束，忽略结构约束，然后比较目标)
            False -> FEA/DA 模式 (考虑总约束，然后比较目标)
    """
    objs = get_obj_tensor(pop)  # [N, M]
    cvs = get_specific_cv(pop, ignore_structural)  # [N]

    N = len(pop)
    dominated = np.zeros(N, dtype=bool)

    # 两两比较
    for i in range(N):
        for j in range(i + 1, N):
            # 比较个体 i 和 j

            # --- CDP 规则 ---
            # 1. 比较约束违背程度 (CV)
            if cvs[i] < cvs[j]:
                # i 的约束更小 -> i 支配 j
                dominated[j] = True
                continue  # 既然关系已定，跳过后续比较
            elif cvs[j] < cvs[i]:
                # j 的约束更小 -> j 支配 i
                dominated[i] = True
                continue

            # 2. 如果 CV 相同 (例如都为0，或者都有相同的化学违背)
            # 则比较目标值 (Pareto Dominance)
            # 只有在 ignore_structural=True 且两者化学均可行时，
            # 才会出现结构差的解有机会支配结构好的解的情况（因为结构被忽略了）

            dom_i_j = np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j])
            dom_j_i = np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i])

            if dom_i_j:
                dominated[j] = True
            if dom_j_i:
                dominated[i] = True

    return ~dominated  # 返回非支配解 (True 表示未被支配)


from copy import deepcopy
import numpy as np
from scipy.spatial.distance import cdist


def truncation(pop, k, custom_objs=None):
    """
    截断选择：移除 k 个最拥挤的个体
    custom_objs: 如果不为None，则基于此矩阵计算距离（用于归一化后的数据）
    """
    if k <= 0:
        return pop

    # 1. 确定用于计算距离的目标矩阵
    if custom_objs is not None:
        objs = custom_objs
    else:
        objs = get_obj_tensor(pop)

    # 2. 计算距离矩阵 (欧氏距离)
    Distance = cdist(objs, objs, metric='euclidean')
    np.fill_diagonal(Distance, np.inf)

    # 3. 迭代删除
    Del = np.zeros(len(pop), dtype=bool)
    while np.sum(Del) < k:
        Remain = np.where(~Del)[0]
        if len(Remain) == 0: break

        # 提取剩余个体的距离子矩阵
        Temp = Distance[np.ix_(Remain, Remain)]

        # 对每行进行排序，找到第k近邻
        # sort axis=1, 得到每行从小到大的距离
        sorted_dist = np.sort(Temp, axis=1)

        # 寻找最拥挤的个体 (Rank 逻辑)
        # 依次比较第1近邻距离, 第2近邻距离...
        # 这里的实现为了效率，通常比较第k近邻 (k=1)
        # 找到第1近邻距离最小的行
        min_dists = sorted_dist[:, 0]
        min_val = np.min(min_dists)

        # 可能有多个个体具有相同的最小距离
        candidates = np.where(min_dists == min_val)[0]

        if len(candidates) > 1:
            # 如果平局，比较第2近邻...以此类推
            # 简化实现：直接删除 candidates 中的第一个
            target_idx_local = candidates[0]
        else:
            target_idx_local = candidates[0]

        # 映射回全局索引并标记删除
        target_idx_global = Remain[target_idx_local]
        Del[target_idx_global] = True

    # 4. 返回保留的个体 (注意：这里不需要deepcopy，因为外面已经copy过了，或者返回引用即可)
    return [pop[i] for i in range(len(pop)) if not Del[i]]


def generate_uniform_points(N, M):
    """简单的参考向量生成 (针对 2D/3D 目标)"""
    if M == 2:
        return np.array([[x / (N - 1), 1 - x / (N - 1)] for x in range(N)]), N
    else:
        # 简化处理，实际高维需要 Das-Dennis 方法
        # 这里仅提供一个伪实现，如果是3目标建议使用 pymoo 的 get_reference_directions
        try:
            from pymoo.util.ref_dirs import get_reference_directions
            ref_dirs = get_reference_directions("das-dennis", M, n_partitions=12)  # 调整 partitions 以接近 N
            return ref_dirs, len(ref_dirs)
        except:
            # 降级：随机生成并归一化
            W = np.random.rand(N, M)
            return W / np.sum(W, axis=1, keepdims=True), N


def update_FA(FA, Offspring, zmin, Ns):
    """
    前向探索存档更新 (Forward Exploration Archive)
    Add=1 (SPEA2 style environmental selection)
    """
    # 1. 深拷贝并合并 (防止后续修改影响存档)
    candidates = [deepcopy(p) for p in FA] + [deepcopy(p) for p in Offspring]

    # 2. 非支配排序 (忽略约束 add=0)
    is_non_dominated = calc_domination(candidates, ignore_structural=True)
    FA_next = [candidates[i] for i in range(len(candidates)) if is_non_dominated[i]]

    if len(FA_next) > Ns:
        # 归一化和截断逻辑...
        PopObj = get_obj_tensor(FA_next)
        zmax = np.max(PopObj, axis=0)
        denom = zmax - zmin
        denom[denom == 0] = 1e-6
        NormObj = (PopObj - zmin) / denom
        FA_next = truncation(FA_next, len(FA_next) - Ns, custom_objs=NormObj)

    return FA_next


def update_FEA(FEA, Offspring, N):
    """
    可行性开发存档 (FEA):
    1. 合并并进行 Constraint-Pareto 非支配排序。
    2. 如果非支配解数量 < N：从被支配解中按 CV 从小到大填充。
    3. 如果非支配解数量 > N：优先保留可行解，并在可行解中进行基于密度的截断。
    """
    # 1. 深拷贝并合并
    candidates = [deepcopy(p) for p in FEA] + [deepcopy(p) for p in Offspring]

    # 2. 计算所有个体的总约束违背 (Total CV)
    # FEA 必须考虑所有约束 (ignore_structural=False)
    all_cvs = get_specific_cv(candidates, ignore_structural=False)

    # 3. 非支配排序 (CDP)
    is_non_dominated = calc_domination(candidates, ignore_structural=False)

    # 分离非支配解(优) 和 被支配解(劣) 的索引
    all_indices = np.arange(len(candidates))
    non_dom_indices = all_indices[is_non_dominated]
    dom_indices = all_indices[~is_non_dominated]

    # 构造初始的 FEA_next (包含所有非支配解)
    FEA_next = [candidates[i] for i in non_dom_indices]
    P = len(FEA_next)

    # 4. 数量判断与填充/截断
    if P <= N:
        # --- Case A: 非支配解不足，需要填充 ---
        needed = N - P
        if needed > 0 and len(dom_indices) > 0:
            # 获取被支配解的 CV 值
            dom_cvs = all_cvs[dom_indices]

            # 按 CV 值从小到大排序
            sorted_rel_idx = np.argsort(dom_cvs)

            # 取前 needed 个 CV 最小的解
            picks = sorted_rel_idx[:needed]

            # 将选中的解加入 FEA_next
            for rel_idx in picks:
                global_idx = dom_indices[rel_idx]
                FEA_next.append(candidates[global_idx])

        return FEA_next

    else:
        # --- Case B: 非支配解过多，需要截断 ---
        # 此时 FEA_next 里的都是非支配解

        # 重新获取 FEA_next 对应的 CV (其实非支配解通常 CV=0 或 CV 很小，但为了严谨)
        # 这里可以直接复用 all_cvs
        current_cvs = all_cvs[non_dom_indices]

        # 统计可行解 (CV <= 0)
        feasible_mask = current_cvs <= 0
        feasible_indices = np.where(feasible_mask)[0]  # 这里的索引是相对于 FEA_next 的

        num_feasible = len(feasible_indices)

        if num_feasible <= N:
            # 情况 B1: 可行解数量不足 N (说明有很多不可行解即使是非支配的)
            # 优先保留所有可行解，剩余位置由不可行解中 CV 最小的填充
            # 或者简单点：直接按 CV 排序保留前 N 个 (因为可行解 CV=0 肯定排在前面)
            sorted_indices = np.argsort(current_cvs)
            FEA_final = [FEA_next[i] for i in sorted_indices[:N]]
            return FEA_final

        else:
            # 情况 B2: 可行解数量 > N
            # 只保留可行解，并进行基于拥挤度的截断 (SPEA2 Truncation)
            FEA_feasible = [FEA_next[i] for i in feasible_indices]

            # 准备归一化数据用于截断
            PopObj = get_obj_tensor(FEA_feasible)
            f_zmin = np.min(PopObj, axis=0)
            f_zmax = np.max(PopObj, axis=0)
            denom = f_zmax - f_zmin
            denom[denom == 0] = 1e-6
            NormObj = (PopObj - f_zmin) / denom

            # 截断删除多余的
            k_del = len(FEA_feasible) - N
            FEA_final = truncation(FEA_feasible, k_del, custom_objs=NormObj)

            return FEA_final


def update_DA(DA, Offspring, zmin, Ns):
    """
    多样性增强存档更新 (Diversity Enhancement Archive)
    严格对应 MATLAB 的 DiversityEnhancementArchive.m
    """
    # 1. 深拷贝并合并 (防止修改源数据)
    candidates = [deepcopy(p) for p in DA] + [deepcopy(p) for p in Offspring]

    # 2. 非支配排序 (Consider Constraints, Add=1)
    is_non_dominated = calc_domination(candidates, ignore_structural=False)
    S = [candidates[i] for i in range(len(candidates)) if is_non_dominated[i]]

    if not S: return []

    # 提取 Obj [N, M] 和 CV [N]
    Obj = get_obj_tensor(S)
    CV = get_specific_cv(S, ignore_structural=False)
    N, M = Obj.shape

    # 3. 生成参考向量 W (UniformPoint)
    # 注意：这里假设 generate_uniform_points 返回 (W, Ns)
    W, _ = generate_uniform_points(Ns, M)

    # 4. 计算 W 自身之间的最小角度 h
    # MATLAB: Angle_W_to_W = acos(1 - pdist2(W,W,'cosine'));
    w_dists = cdist(W, W, 'cosine')
    # 限制数值范围防止 nan (cos值在 [0, 1] 之间，dist 在 [0, 2])
    # 理论上 dist 应该在 [0, 1] 之间如果向量是正的
    w_dists = np.clip(w_dists, 0, 2)
    w_angles = np.arccos(1 - w_dists)

    # 对角线设为无穷大，为了找非自身的最小值
    np.fill_diagonal(w_angles, np.inf)
    h = np.mean(np.min(w_angles, axis=1))

    # 5. 归一化 (Environmental selection part)
    # MATLAB:
    # if sum(CV <= 0) > 0
    #    zmax = max(Obj(CV <= 0, :), [], 1);
    # else ...
    feasible_mask_global = CV <= 0
    if np.any(feasible_mask_global):
        zmax = np.max(Obj[feasible_mask_global], axis=0)
    else:
        # 如果全是不可行解，找 CV 最小的那个对应的 Obj 作为 zmax
        idx = np.argmin(CV)
        zmax = Obj[idx]

    denom = zmax - zmin
    denom[denom == 0] = 1e-6  # 防止除零
    Obj_norm = (Obj - zmin) / denom

    # 6. 计算 S 到 W 的角度 (正弦值)
    # MATLAB: Angle_S_to_W = sin(acos(1 - pdist2(W,Obj,'cosine')));
    s_w_dists = cdist(W, Obj_norm, 'cosine')
    # 限制范围，防止 acos 出错
    s_w_dists = np.clip(s_w_dists, 0, 2)
    # 这里的 1-dist 实际上是 cosine similarity
    s_w_angles = np.arccos(1 - s_w_dists)
    Angle_S_to_W = np.sin(s_w_angles)  # Shape: [Ns, len(S)]

    DA_next = []

    # 7. 为每个参考向量选择一个个体
    for i in range(Ns):
        Angle = Angle_S_to_W[i, :]  # 当前参考向量 i 到所有个体的角度

        # 筛选角度小于 h 的候选者
        # list = Angle <= h;
        candidates_mask = Angle <= h

        # 如果没有候选者，选角度最小的那个
        if np.sum(candidates_mask) == 0:
            min_angle_idx = np.argmin(Angle)
            candidates_mask[min_angle_idx] = True

        # --- 核心选择逻辑 (完全复刻 MATLAB T向量逻辑) ---
        # T = inf(length(S), 1);
        T = np.full(N, np.inf)

        # T(list) = CV(list);
        # 将候选者的 CV 值填入 T，非候选者保持 inf
        T[candidates_mask] = CV[candidates_mask]

        # feasible = T <= 0;
        # 这里 T<=0 意味着：1. 是候选者(mask为True) 2. 且 CV<=0
        feasible_candidates_mask = T <= 0

        # if sum(feasible) <= 0
        if np.sum(feasible_candidates_mask) == 0:
            # 没有可行的候选解：选择 T (即 CV) 最小的
            # [~,index] = min(T);
            best_idx = np.argmin(T)
        else:
            # 存在可行的候选解：基于角度选择
            # T = inf(size(Angle));
            T[:] = np.inf

            # Fitness = Angle_S_to_W(i,:);
            # T(feasible) = Fitness(feasible);
            # 只有那些既在候选区又可行的解，才填入角度值
            T[feasible_candidates_mask] = Angle[feasible_candidates_mask]

            # [~,index] = min(T);
            best_idx = np.argmin(T)

        DA_next.append(S[best_idx])

    DA_next = evo.decouple_memory_references(DA_next)
    return DA_next