from rdkit import Chem

from qm9.rdkit_functions import build_xae_molecule

# 定义键类型字典 (适配你的 build_molecule)
bond_dict = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC
}


def build_molecule_with_3d(positions, atom_types, dataset_info):
    """
    基于你提供的函数升级：不仅构建拓扑图，还注入 3D 坐标，这是计算 USRCAT 的前提！
    """
    atom_decoder = dataset_info["atom_decoder"]

    # 假设 build_xae_molecule 存在于你的当前命名空间中
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)

    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        # 防止重复添加双向边
        if bond[0].item() > bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])

    mol = mol.GetMol()

    # 【关键新增】：注入 3D 坐标
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        # 将 tensor 坐标写入
        conf.SetAtomPosition(i, (positions[i, 0].item(), positions[i, 1].item(), positions[i, 2].item()))
    mol.AddConformer(conf)

    # 尝试消毒，失败也不影响我们提取基础特征
    try:
        Chem.SanitizeMol(mol)
    except:
        pass

    return mol


def get_rdkit_structural_features(pop, dataset_info):
    """
    分别提取 2D 和 3D 特征，解决维度冲突和距离尺度不一的问题。
    返回: (features_2d[N, 2048], features_3d [N, 60])
    """
    import torch
    import numpy as np
    from rdkit.Chem import rdMolDescriptors
    from rdkit import DataStructs

    fp_features = []
    usr_features = []

    # 统一正确的维度
    default_fp = np.zeros(2048)
    default_usr = np.zeros(60)  # USRCAT 严格是 60 维

    for p in pop:
        fp_feat = default_fp
        usr_feat = default_usr
        try:
            n_atoms = p['long']
            positions = p['x'][:n_atoms].detach().cpu()
            atom_types = p['atom_type'][:n_atoms].detach().cpu()

            # 使用你之前定义的 build_molecule_with_3d
            mol = build_molecule_with_3d(positions, atom_types, dataset_info)

            if mol is not None:
                # 1. 2D 拓扑特征 (Morgan)
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fp_feat = arr

                # 2. 3D 形状特征 (USRCAT)
                if mol.GetNumConformers() > 0:
                    usr_feat = np.array(rdMolDescriptors.GetUSRCAT(mol))

        except Exception:
            pass  # 遇到烂分子直接使用全 0 特征

        fp_features.append(fp_feat)
        usr_features.append(usr_feat)

    return np.array(fp_features, dtype=np.float32), np.array(usr_features, dtype=np.float32)


from scipy.spatial.distance import cdist


def calc_mixed_structure_distance_matrix(feat2d_A, feat3d_A, feat2d_B=None, feat3d_B=None):
    """
    计算 A 和 B 之间的混合结构距离矩阵。
    2D 使用 Jaccard 距离，3D 使用 Euclidean 距离。
    """
    if feat2d_B is None:
        feat2d_B = feat2d_A
        feat3d_B = feat3d_A

    # 1. 2D Tanimoto/Jaccard 距离 (值域 0~1)
    # cdist 中 jaccard 距离定义为 1 - (交集/并集)，正适合 0/1 向量
    dist_2d = cdist(feat2d_A, feat2d_B, metric='jaccard')
    # 处理全 0 向量导致的 nan
    dist_2d = np.nan_to_num(dist_2d, nan=1.0)

    # 2. 3D 欧氏距离
    dist_3d = cdist(feat3d_A, feat3d_B, metric='euclidean')
    # 归一化 3D 距离到 0~1 左右，使其与 2D 距离处于同一量级
    max_3d = np.max(dist_3d)
    if max_3d > 0:
        dist_3d = dist_3d / max_3d

    # 3. 混合距离 (权重可调，这里各占 50%)
    mixed_dist = 0.5 * dist_2d + 0.5 * dist_3d
    return mixed_dist





import torch


@torch.no_grad()
def Get_Fitness_Pareto_Main(Pop, dataset_info, device, preds, n_nodes, ObjName, patt, tolerance, minimize=True,
                            num_workers=1):
    """
    实际上是评价分子，懒得改了
    """
    # 1. 基础评价
    if num_workers > 1:
        # 这里假设你实现了多进程的 EvalPop_Patt_MP
        Pop = evo.EvalPop_Patt_MP(Pop, patt, dataset_info, tolerance, num_workers=num_workers)
    else:
        Pop = evo.EvalPop_Patt(Pop, patt, dataset_info, tolerance)

    Pop = evo.EvalPop_Con(Pop, dataset_info)
    Pop = evo.EvalPop_Obj(Pop, device, preds, n_nodes)

    return Pop


def FNDS_StructDiv(PopObj, PopStruct_2D, PopStruct_3D, PopCon=None):
    """
    接收分离的 2D 和 3D 特征的 FNDS
    """
    N = PopObj.shape[0]
    CV = np.zeros(N) if PopCon is None else np.maximum(0, PopCon)

    # 1. 目标空间支配计算 (保持不变)
    Dominate = np.zeros((N, N), dtype=bool)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if CV[i] < CV[j]:
                Dominate[i, j] = True
            elif CV[i] > CV[j]:
                Dominate[j, i] = True
            else:
                better_i = np.any(PopObj[i] < PopObj[j]) and np.all(PopObj[i] <= PopObj[j])
                better_j = np.any(PopObj[j] < PopObj[i]) and np.all(PopObj[j] <= PopObj[i])
                if better_i:
                    Dominate[i, j] = True
                elif better_j:
                    Dominate[j, i] = True

    S = np.sum(Dominate, axis=1)
    R = np.zeros(N)
    for i in range(N): R[i] = np.sum(S[Dominate[:, i]])

    # 2. 化学结构空间拥挤度计算 (使用混合距离)
    Distance = calc_mixed_structure_distance_matrix(PopStruct_2D, PopStruct_3D)
    np.fill_diagonal(Distance, np.inf)
    Distance.sort(axis=1)

    k = max(1, int(np.sqrt(N)))
    D = 1 / (Distance[:, k - 1] + 2)

    Fitness = R + D
    return Fitness




def EnvironmentalSelectionMain_Diverse(Pop, N, ObjName, dataset_info, min_dist_threshold=0.5):
    """
    SAES
    """
    Num_Pop = len(Pop)
    if Num_Pop <= N:
        return Pop

    # 1. 提取 Obj 和 Con
    Obj = np.array([[p[obj].item() if torch.is_tensor(p[obj]) else p[obj] for obj in ObjName] for p in Pop])

    Con = []
    for p in Pop:
        c1 = max(0, p.get('Constraint', 0))
        c2 = sum(p.get('structure_Con', [0])) if isinstance(p.get('structure_Con', 0), list) else p.get('structure_Con',
                                                                                                        0)
        Con.append(c1 + max(0, c2))
    CV = np.array(Con)

    # 2. 提取结构特征并计算距离矩阵 (复用之前的函数)
    Struct_2D, Struct_3D = get_rdkit_structural_features(Pop, dataset_info)
    Dist = calc_mixed_structure_distance_matrix(Struct_2D, Struct_3D)

    # 3. 计算 CDP 支配强度 R (类似 SPEA2 的 Rank，越小越好，0代表非支配)
    Dominate = np.zeros((Num_Pop, Num_Pop), dtype=bool)
    for i in range(Num_Pop):
        for j in range(Num_Pop):
            if i == j: continue
            # CDP 规则
            if CV[i] < CV[j]:
                Dominate[i, j] = True
            elif CV[i] == CV[j]:
                better_i = np.any(Obj[i] < Obj[j]) and np.all(Obj[i] <= Obj[j])
                if better_i:
                    Dominate[i, j] = True

    S = np.sum(Dominate, axis=1)  # 个体 i 支配了多少人
    R = np.zeros(Num_Pop)
    for i in range(Num_Pop):
        R[i] = np.sum(S[Dominate[:, i]])  # 支配个体 i 的人的 S 之和

    # =======================================================
    # 4. 多样性优先的贪心选择算法 (Diversity-First Greedy Selection)
    # =======================================================
    selected_idx = []
    unselected_idx = list(range(Num_Pop))

    # 第一步：选出第一颗“种子”
    # 在最优层(Rank最小)中，找一个与其它人平均距离最远的分子（最具代表性）
    min_R = np.min(R)
    candidates = [i for i in unselected_idx if R[i] == min_R]
    if len(candidates) == 1:
        first_idx = candidates[0]
    else:
        avg_dists = np.mean(Dist[candidates, :], axis=1)
        first_idx = candidates[np.argmax(avg_dists)]

    selected_idx.append(first_idx)
    unselected_idx.remove(first_idx)

    # 第二步：迭代选择剩余的 N-1 个位置
    while len(selected_idx) < N and len(unselected_idx) > 0:
        # 计算所有未被选中的人，到【已选中集合】的最短结构距离
        d_to_S = np.min(Dist[unselected_idx][:, selected_idx], axis=1)

        # 核心逻辑：找出结构差异达标的候选者 (杜绝克隆体)
        valid_mask = d_to_S >= min_dist_threshold

        if np.any(valid_mask):
            # 场景 A: 存在结构达标的分子
            valid_indices = np.array(unselected_idx)[valid_mask]
            valid_R = R[valid_indices]
            valid_d = d_to_S[valid_mask]

            # 在达标的分子中，优先选 Rank 最好的 (逼近 Pareto 前沿)
            best_valid_R = np.min(valid_R)
            best_R_mask = (valid_R == best_valid_R)
            best_R_indices = valid_indices[best_R_mask]
            best_R_d = valid_d[best_R_mask]

            # 如果有多个人 Rank 一样好，选离已选集合最远的 (MaxMin Diversity)
            target_idx = best_R_indices[np.argmax(best_R_d)]
        else:
            # 场景 B: 妥协。所有剩下的分子都跟已选中的分子长得很像 (d < threshold)
            # 说明化学空间探索枯竭。此时放弃多样性阈值，优先选 Rank 好的。
            min_R_unselected = np.min(R[unselected_idx])
            fallback_indices = [i for i in unselected_idx if R[i] == min_R_unselected]
            if len(fallback_indices) == 1:
                target_idx = fallback_indices[0]
            else:
                # 同 Rank 中依然选距离相对最远的
                d_subset = np.min(Dist[fallback_indices][:, selected_idx], axis=1)
                target_idx = fallback_indices[np.argmax(d_subset)]

        selected_idx.append(target_idx)
        unselected_idx.remove(target_idx)

    # 5. 写入 Fitness 供外层调用统一兼容
    # 选中的人 fitness < 1 (越早选中的值越小)
    for rank_pos, idx in enumerate(selected_idx):
        Pop[idx]['fitness'] = rank_pos / N
        # 没选中的人 fitness > 1
    for idx in unselected_idx:
        Pop[idx]['fitness'] = 1.0 + R[idx]

    return [Pop[i] for i in selected_idx]


from scipy.spatial.distance import cdist





import numpy as np
import torch
import evo







def _diverse_truncation(Pop, Fitness, dataset_info, N, min_dist=0.05):
    """通用底层的SAES"""
    Num_Pop = len(Pop)
    if Num_Pop <= N:
        for i, p in enumerate(Pop): p['fitness'] = Fitness[i]
        return Pop

    Struct_2D, Struct_3D = get_rdkit_structural_features(Pop, dataset_info)
    Dist = calc_mixed_structure_distance_matrix(Struct_2D, Struct_3D)

    selected_idx = []
    unselected_idx = list(range(Num_Pop))

    # 1. 挑第一个种子 (Fitness 最好中，距离其他最远的)
    min_fit = np.min(Fitness)
    candidates = [i for i in unselected_idx if Fitness[i] == min_fit]
    first_idx = candidates[np.argmax(np.mean(Dist[candidates, :], axis=1))] if len(candidates) > 1 else candidates[0]

    selected_idx.append(first_idx)
    unselected_idx.remove(first_idx)

    # 2. 迭代挑选
    while len(selected_idx) < N:
        d_to_S = np.min(Dist[unselected_idx][:, selected_idx], axis=1)
        valid_mask = d_to_S >= min_dist

        if np.any(valid_mask):
            valid_indices = np.array(unselected_idx)[valid_mask]
            valid_fits = Fitness[valid_indices]
            best_fit = np.min(valid_fits)
            best_indices = valid_indices[valid_fits == best_fit]
            target_idx = best_indices[np.argmax(d_to_S[valid_mask][valid_fits == best_fit])]
        else:
            # 妥协，选 Fitness 最好
            min_fit_unselected = np.min(Fitness[unselected_idx])
            fallback_indices = [i for i in unselected_idx if Fitness[i] == min_fit_unselected]
            d_subset = np.min(Dist[fallback_indices][:, selected_idx], axis=1)
            target_idx = fallback_indices[np.argmax(d_subset)]

        selected_idx.append(target_idx)
        unselected_idx.remove(target_idx)

    selected_pop = [Pop[i] for i in selected_idx]
    for rank_pos, idx in enumerate(selected_idx):
        selected_pop[rank_pos]['fitness'] = rank_pos / N  # 重新赋值 rank

    return selected_pop



def EnvironmentalSelection_A(Pop, N, ObjName, dataset_info):
    """
    种群 A (底盘探索者)：
    目标：优化 Vina，且尽可能【远离】目标片段 (最大化 structure_Con)。
    机制：FNDS_StructDiv + _diverse_truncation 双重保证内部结构多样性。
    """
    Objs = []
    ChemCons = []
    for p in Pop:
        obj_vals = [p[obj].item() if torch.is_tensor(p[obj]) else p[obj] for obj in ObjName]

        s_con = p.get('structure_Con', 0)
        s_sum = sum(s_con) if isinstance(s_con, list) else s_con

        # 核心：最小化 -s_sum，迫使 A 种群寻找完全不包含目标片段的奇异骨架
        obj_vals.append(-s_sum)
        Objs.append(obj_vals)
        # 化学稳定性依然是硬底线
        ChemCons.append(max(0, p.get('Constraint', 0)))

    Objs = np.array(Objs)
    Struct_2D, Struct_3D = get_rdkit_structural_features(Pop, dataset_info)

    Fitness = FNDS_StructDiv(Objs, Struct_2D, Struct_3D, PopCon=np.array(ChemCons))

    # 强制内部结构截断，绝不允许 A 种群出现克隆体
    return _diverse_truncation(Pop, Fitness, dataset_info, N, min_dist=0.05)


def EnvironmentalSelection_B(Pop, N, ObjName, dataset_info):
    """
    种群 B (局部打磨者)：
    严格限制：只收留【不可行解】（化学违背 > 0 或 结构违背 > 0）。
    目标：优化 Vina，且尽可能【拼上】目标片段 (最小化 structure_Con)。
    """
    # 1. 严格过滤：只保留半成品 (Infeasible Only)
    pool = []
    for p in Pop:
        c1 = max(0, p.get('Constraint', 0))
        s_con = p.get('structure_Con', 0)
        c2 = sum(s_con) if isinstance(s_con, list) else s_con

        if max(0, c2) > 0:  # 只要有一丁点不合规，就收入 B 车间打磨
            pool.append(p)

    # 极端情况兜底：如果全宇宙都完美合规了，随便塞点进去维持矩阵运转
    if len(pool) == 0:
        pool = Pop

        # 2. 计算适应度
    Objs = []
    ChemCons = []
    for p in pool:
        obj_vals = [p[obj].item() if torch.is_tensor(p[obj]) else p[obj] for obj in ObjName]
        s_con = p.get('structure_Con', 0)
        s_sum = sum(s_con) if isinstance(s_con, list) else s_con

        # B 的目标是把 s_sum 降到 0
        obj_vals.append(s_sum)
        Objs.append(obj_vals)
        ChemCons.append(max(0, p.get('Constraint', 0)))

    Objs = np.array(Objs)
    Struct_2D, Struct_3D = get_rdkit_structural_features(pool, dataset_info)

    Fitness = FNDS_StructDiv(Objs, Struct_2D, Struct_3D, PopCon=np.array(ChemCons))

    # 强制 B 种群内部也必须维持结构多样性
    return _diverse_truncation(pool, Fitness, dataset_info, min(N, len(pool)), min_dist=0.05)


def EnvironmentalSelection_C_Archive(Pop, N, ObjName, dataset_info):
    """
    种群 C (精英陈列室)：
    严格限制：只收留【完美可行解】（Total CV == 0）。
    """
    feasible_pop = []
    for p in Pop:
        c1 = max(0, p.get('Constraint', 0))
        s_con = p.get('structure_Con', 0)
        c2 = sum(s_con) if isinstance(s_con, list) else s_con
        if c1 + max(0, c2) <= 0:
            feasible_pop.append(p)

    if len(feasible_pop) == 0: return []
    if len(feasible_pop) <= N: return feasible_pop

    Objs = np.array([[p[obj].item() if torch.is_tensor(p[obj]) else p[obj] for obj in ObjName] for p in feasible_pop])
    Struct_2D, Struct_3D = get_rdkit_structural_features(feasible_pop, dataset_info)

    Fitness = FNDS_StructDiv(Objs, Struct_2D, Struct_3D, PopCon=np.zeros(len(feasible_pop)))
    return _diverse_truncation(feasible_pop, Fitness, dataset_info, N, min_dist=0.05)




