from qm9.sampling import sample_chain, sample, sample_sweep_conditional,sample_max_n
from equivariant_diffusion import utils as diffusion_utils
from qm9 import dataset,analyze
from qm9.property_prediction.models_property import EGNN
from qm9.property_prediction import prop_utils
import copy
import itertools
import math
from qm9.visualizer import plot_data3d,save_xyz_file,load_molecule_xyz
from scipy.spatial.distance import pdist, squareform
#matplotlib.use('TkAgg')
from qm9.bond_analyze import get_bond_order
from rdkit.Chem import rdMolDescriptors, Lipinski, Crippen
import torch.nn.functional as F
import concurrent.futures
from collections import deque
import random
try:
    from posebusters import PoseBusters
    import openbabel
    from tblite.ase import TBLite as ASETBLite  # 用于优化
    from xtb.interface import Calculator as XtbCalculator, Param  # 用于算属性
    from sklearn.metrics import r2_score
    from xtb.interface import Calculator, Param
    from xtb.libxtb import VERBOSITY_MUTED
    from tblite.ase import TBLite
    from analysis import analyze
    from ase import Atoms
    from ase.optimize import BFGS
except ImportError:
    pass  # 后面代码需确保 analyze 可用
from scipy.stats import pearsonr
from contextlib import contextmanager
import sys
import torch
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from copy import deepcopy
import os
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski, AllChem,QED
from rdkit import DataStructs
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.error')
#这个是评价器的mae和mad
property_mean = {'alpha' :torch.tensor(75.3734),'gap' :torch.tensor(6.8632),'homo' :torch.tensor(-6.5386),'lumo' :torch.tensor(0.3246),'mu' :torch.tensor(2.6751),'Cv' :torch.tensor(-22.1666)}
property_mad = {'alpha' : torch.tensor(6.2728),'gap' :torch.tensor(1.0619),'homo' :torch.tensor(0.4396),'lumo' :torch.tensor(1.0338),'mu' :torch.tensor(1.1757),'Cv' :torch.tensor(4.8885)}
conversion = {'alpha' :torch.tensor(1),'gap' :torch.tensor(1000),'homo' :torch.tensor(1000),'lumo' :torch.tensor(1000),'mu' :torch.tensor(1),'Cv' :torch.tensor(1)}
#HAR2EV = 27.211386246
#KCALMOL2EV = 0.04336414

# def sdf_to_pdbqt(sdf_file, pdbqt_outfile, conda_bin_dir=""):
#     os.environ['BABEL_LIBDIR'] = '/your_conda_env_dir/lib/openbabel/3.1.0/' #if linux
#     obabel_exec_path = os.path.join(conda_bin_dir, "obabel")
#     cmd = [obabel_exec_path, sdf_file, '-O', pdbqt_outfile]
#     subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, check=False)

import os
import sys
import shutil
import subprocess

def sdf_to_pdbqt(sdf_file, pdbqt_outfile, conda_bin_dir=""):
    """
    全平台自适应的 sdf 转 pdbqt 函数，自动寻找当前 Conda 环境下的 obabel 和 依赖库
    """


    env_root = sys.prefix

    obabel_exec_path = None
    babel_base_libdir = None

    # 1. 根据操作系统，拼接可能存在的路径
    if os.name == 'nt':  # Windows 系统
        potential_exec_paths =[
            os.path.join(env_root, "Library", "bin", "obabel.exe"),  # 大多数 C/C++ 编译包的存放处
            os.path.join(env_root, "Scripts", "obabel.exe"),
            shutil.which("obabel.exe"),
            shutil.which("obabel")
        ]
        # Windows 下的库文件夹通常在 Library\lib 下
        babel_base_libdir = os.path.join(env_root, "Library", "lib", "openbabel")
    else:  # Linux / Mac 系统
        potential_exec_paths =[
            os.path.join(env_root, "bin", "obabel"),
            shutil.which("obabel")
        ]
        babel_base_libdir = os.path.join(env_root, "lib", "openbabel")

    # 2. 遍历寻找可执行文件
    for p in potential_exec_paths:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            obabel_exec_path = p
            break

    if not obabel_exec_path:
        raise FileNotFoundError(f"Error: Could not find 'obabel' in {env_root} or system PATH.")

    # 3. 动态配置 BABEL_LIBDIR 插件目录 (解决 format 转换报错问题)
    if os.path.exists(babel_base_libdir):
        # 遍历拿到类似 3.1.0, 3.1.1 这样的版本号文件夹
        subdirs =[os.path.join(babel_base_libdir, d) for d in os.listdir(babel_base_libdir)
                   if os.path.isdir(os.path.join(babel_base_libdir, d))]
        if subdirs:
            os.environ['BABEL_LIBDIR'] = subdirs[0]

    # 4. 组装并执行命令
    cmd = [obabel_exec_path, sdf_file, '-O', pdbqt_outfile]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"OpenBabel conversion failed: {e}")

def get_bond_mat(coords, atom_types):
    """
    向量化优化的邻接矩阵构建
    """
    num_nodes = coords.shape[0]
    if num_nodes == 0:
        return np.zeros((0, 0), dtype=int)

    # 利用广播机制计算欧氏距离矩阵 [N, N]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    # 仅遍历上三角以减少一半的 get_bond_order 调用
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            order = get_bond_order(atom_types[i], atom_types[j], dist_matrix[i, j])
            if order > 0:
                adj[i, j] = adj[j, i] = order
    return adj

def build_mol_from_coords_and_types(coords, atom_types):
    int2bond = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC
    }
    mol = Chem.RWMol()
    atom_indices = [mol.AddAtom(Chem.Atom(t)) for t in atom_types]

    coords_np = coords.numpy() if torch.is_tensor(coords) else coords
    num_atoms = len(atom_types)

    # 优化：预先计算距离减少重复开销
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(coords_np[i] - coords_np[j])
            bond_int = get_bond_order(atom_types[i], atom_types[j], dist)
            bond_type = int2bond.get(bond_int)
            if bond_type:
                mol.AddBond(atom_indices[i], atom_indices[j], bond_type)

    try:
        res_mol = mol.GetMol()
        Chem.SanitizeMol(res_mol)
        return res_mol
    except:
        return mol.GetMol()


def _is_connected_fragment_subset(adj_matrix, nodes):
    """Return whether ``nodes`` induce one connected fragment subgraph."""
    if len(nodes) <= 1:
        return True
    allowed = set(nodes)
    visited = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        current = stack.pop()
        for neighbor in np.where(adj_matrix[current] != 0)[0]:
            neighbor = int(neighbor)
            if neighbor in allowed and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(allowed)


def _enumerate_subgraph_embeddings(
        frag_adj,
        frag_types,
        target_adj,
        target_types,
        fragment_nodes,
        blacklist=None,
):
    """Enumerate unique target atom sets embedding an induced fragment subset.

    The embedding is a non-induced subgraph monomorphism: every fragment bond
    and bond order must be present in the target, while extra target bonds are
    allowed.  Fragment traversal order is chosen from graph selectivity rather
    than atom indices, making the result invariant to atom renumbering.
    """
    fragment_nodes = tuple(int(node) for node in fragment_nodes)
    blocked = set() if blacklist is None else {int(node) for node in blacklist}
    if not fragment_nodes:
        return []

    selected = set(fragment_nodes)
    fragment_degrees = {
        node: sum(
            1
            for neighbor in selected
            if neighbor != node and frag_adj[node, neighbor] != 0
        )
        for node in fragment_nodes
    }
    target_degrees = np.count_nonzero(target_adj, axis=1)
    candidates = {}
    for fragment_node in fragment_nodes:
        candidates[fragment_node] = [
            target_node
            for target_node, target_type in enumerate(target_types)
            if target_node not in blocked
            and target_type == frag_types[fragment_node]
            and target_degrees[target_node] >= fragment_degrees[fragment_node]
        ]
        if not candidates[fragment_node]:
            return []

    order = sorted(
        fragment_nodes,
        key=lambda node: (
            len(candidates[node]),
            -fragment_degrees[node],
            frag_types[node],
            node,
        ),
    )
    mapping = {}
    used_targets = set()
    embeddings = []
    seen_target_sets = set()

    def has_forward_candidate(fragment_node, target_node):
        for fragment_neighbor in selected:
            required_bond = frag_adj[fragment_node, fragment_neighbor]
            if (
                fragment_neighbor in mapping
                or required_bond == 0
            ):
                continue
            if not any(
                candidate not in used_targets
                and candidate != target_node
                and target_types[candidate] == frag_types[fragment_neighbor]
                and target_adj[target_node, candidate] == required_bond
                for candidate in candidates[fragment_neighbor]
            ):
                return False
        return True

    def backtrack(position):
        if position == len(order):
            target_set = frozenset(mapping.values())
            if target_set not in seen_target_sets:
                seen_target_sets.add(target_set)
                embeddings.append(mapping.copy())
            return

        fragment_node = order[position]
        for target_node in candidates[fragment_node]:
            if target_node in used_targets:
                continue
            feasible = True
            for mapped_fragment, mapped_target in mapping.items():
                required_bond = frag_adj[fragment_node, mapped_fragment]
                if (
                    required_bond != 0
                    and target_adj[target_node, mapped_target] != required_bond
                ):
                    feasible = False
                    break
            if not feasible or not has_forward_candidate(fragment_node, target_node):
                continue

            mapping[fragment_node] = target_node
            used_targets.add(target_node)
            backtrack(position + 1)
            used_targets.remove(target_node)
            del mapping[fragment_node]

    backtrack(0)
    return embeddings


def _full_match_options(
        frag_adj,
        frag_types,
        target_adj,
        target_types,
        blacklist=None,
):
    """Return all unique complete embeddings as ``(size, atoms, mapping)``."""
    fragment_nodes = tuple(range(len(frag_types)))
    embeddings = _enumerate_subgraph_embeddings(
        frag_adj,
        frag_types,
        target_adj,
        target_types,
        fragment_nodes,
        blacklist=blacklist,
    )
    return [
        (len(fragment_nodes), frozenset(mapping.values()), mapping)
        for mapping in embeddings
    ]


def _connected_match_options(
        frag_adj,
        frag_types,
        target_adj,
        target_types,
        blacklist=None,
        full_options=None,
):
    """Enumerate permutation-invariant connected common-subgraph options."""
    fragment_size = len(frag_types)
    if fragment_size == 0:
        return []

    options_by_target_set = {}
    if full_options is None:
        full_options = _full_match_options(
            frag_adj,
            frag_types,
            target_adj,
            target_types,
            blacklist=blacklist,
        )
    for option in full_options:
        options_by_target_set[option[1]] = option

    fragment_indices = tuple(range(fragment_size))
    for size in range(fragment_size - 1, 0, -1):
        for subset in itertools.combinations(fragment_indices, size):
            if not _is_connected_fragment_subset(frag_adj, subset):
                continue
            for mapping in _enumerate_subgraph_embeddings(
                frag_adj,
                frag_types,
                target_adj,
                target_types,
                subset,
                blacklist=blacklist,
            ):
                target_set = frozenset(mapping.values())
                previous = options_by_target_set.get(target_set)
                if previous is None or size > previous[0]:
                    options_by_target_set[target_set] = (
                        size,
                        target_set,
                        mapping,
                    )

    return sorted(
        options_by_target_set.values(),
        key=lambda option: (-option[0], tuple(sorted(option[1]))),
    )


def _choose_disjoint_options(option_lists, fragment_sizes):
    """Choose one non-overlapping option per pattern with maximum coverage."""
    if not option_lists or any(not options for options in option_lists):
        return None

    pattern_order = sorted(
        range(len(option_lists)),
        key=lambda index: (len(option_lists[index]), index),
    )
    max_quality = [
        max(option[0] for option in option_lists[index]) / fragment_sizes[index]
        for index in pattern_order
    ]
    remaining_quality = [0.0] * (len(pattern_order) + 1)
    for position in range(len(pattern_order) - 1, -1, -1):
        remaining_quality[position] = (
            remaining_quality[position + 1] + max_quality[position]
        )

    choices = [None] * len(option_lists)
    best_choices = None
    best_quality = -1.0

    def search(position, used_atoms, quality):
        nonlocal best_choices, best_quality
        # Once the theoretical upper bound is reached, no alternative branch
        # can improve the assignment.  This is especially important for
        # symmetric molecules, which may have many equivalent embeddings.
        if best_quality >= remaining_quality[0] - 1e-12:
            return
        if quality + remaining_quality[position] < best_quality - 1e-12:
            return
        if position == len(pattern_order):
            if quality > best_quality + 1e-12:
                best_quality = quality
                best_choices = list(choices)
            return

        pattern_index = pattern_order[position]
        fragment_size = fragment_sizes[pattern_index]
        for option in option_lists[pattern_index]:
            if not option[1].isdisjoint(used_atoms):
                continue
            choices[pattern_index] = option
            search(
                position + 1,
                used_atoms | option[1],
                quality + option[0] / fragment_size,
            )
            choices[pattern_index] = None

    search(0, frozenset(), 0.0)
    return best_choices


class OptimizedVF2:
    """Compatibility wrapper around the permutation-invariant matcher."""

    def __init__(self, frag_adj, frag_types, target_adj, target_types, blacklist=None):
        self.f_adj, self.f_types = frag_adj, frag_types
        self.t_adj, self.t_types = target_adj, target_types
        self.f_size = len(frag_types)
        self.t_size = len(target_types)
        self.blacklist = set() if blacklist is None else set(blacklist)
        self.max_matched = 0
        self.best_mapping = {}

    def match(self):
        options = _connected_match_options(
            self.f_adj,
            self.f_types,
            self.t_adj,
            self.t_types,
            blacklist=self.blacklist,
        )
        if options:
            self.max_matched = options[0][0]
            self.best_mapping = options[0][2]
        return self.max_matched == self.f_size


def check_subgraph_proportionVF2(dataset_info, fragment_x, fragment_atom_type, fragment_mask,
                                 target_x, target_atom_type, target_mask, blacklist=None):
    f_valid_indices = torch.nonzero(fragment_mask.squeeze(-1)).flatten().cpu().numpy()
    t_valid_indices = torch.nonzero(target_mask.squeeze(-1)).flatten().cpu().numpy()

    f_coords = fragment_x[f_valid_indices].detach().cpu().numpy()
    f_types_idx = fragment_atom_type[f_valid_indices].detach().cpu().numpy()
    t_coords = target_x[t_valid_indices].detach().cpu().numpy()
    t_types_idx = target_atom_type[t_valid_indices].detach().cpu().numpy()

    f_types = [dataset_info['atom_decoder'][i] for i in f_types_idx]
    t_types = [dataset_info['atom_decoder'][i] for i in t_types_idx]

    if not f_types: return 0.0, []

    f_adj = get_bond_mat(f_coords, f_types)
    t_adj = get_bond_mat(t_coords, t_types)

    # [新增] 将全局黑名单转为局部黑名单
    local_blacklist = set()
    if blacklist:
        for local_idx, global_idx in enumerate(t_valid_indices):
            if global_idx in blacklist:
                local_blacklist.add(local_idx)

    # 传入黑名单
    solver = OptimizedVF2(f_adj, f_types, t_adj, t_types, blacklist=local_blacklist)
    solver.match()

    proportion = (solver.max_matched / len(f_types)) * 100
    matched_target_indices = []
    if solver.best_mapping:
        target_local_indices = list(solver.best_mapping.values())
        matched_target_indices = t_valid_indices[target_local_indices].tolist()

    return proportion, matched_target_indices


def check_subgraph_proportionVF2_optimized(dataset_info, fragment_x, fragment_atom_type, fragment_mask,
                                           target_x, target_atom_type, target_mask, blacklist=None):
    # Keep the historical public entry point, but use exactly the same
    # permutation-invariant score and mapping semantics as the reference path.
    return check_subgraph_proportionVF2(
        dataset_info,
        fragment_x,
        fragment_atom_type,
        fragment_mask,
        target_x,
        target_atom_type,
        target_mask,
        blacklist=blacklist,
    )


def read_mol(file,dataset_info,device):
    pos, oneh, _ = load_molecule_xyz(file,dataset_info)
    n = pos.size(0)  # 当前长度
    target_length = 29

    pad_length = target_length - n

    if pad_length > 0:
        pos_padded = F.pad(pos, (0, 0, 0, pad_length))  # 在行末填充0
        oneh_padded = F.pad(oneh, (0, 0, 0, pad_length))  # 在行末填充0
    else:
        pos_padded = pos
        oneh_padded = oneh

    # 创建 node_mask
    node_mask = torch.cat([torch.ones(n, 1), torch.zeros(pad_length, 1)])
    mol = {}
    mol['node_mask'] = node_mask.to(device)
    mol['x'] = pos_padded.to(device)
    mol['one_hot'] = oneh_padded.to(device)
    mol['atom_type'] = torch.argmax(oneh_padded, dim=1).to(device)
    return mol

def save_pop(Pop, output_path, name, dataset_info):
    for idx, data in enumerate(Pop):
        # 获取张量数据并扩展为批量大小为1
        one_hot = data['one_hot'].unsqueeze(0)  # 形状变为 (1, 29, 5)
        positions = data['x'].unsqueeze(0)  # 形状变为 (1, 29, 3)
        node_mask = data['node_mask'].unsqueeze(0)  # 形状变为 (1, 29, 1)

        # 调用函数保存文件
        save_xyz_file(
            path=output_path,
            one_hot=one_hot,
            charges=None,  # 如果不需要，可以传递None
            positions=positions,
            dataset_info=dataset_info,
            id_from=idx,  # 每个分子的初始编号
            name=name,  # 文件名前缀
            node_mask=node_mask
        )

def save3D(molecule,dataset_info,type,index):
    x = molecule['x']
    long = molecule['long']
    atom_type = molecule.get('atom_type', math.nan).detach().cpu() if molecule.get('atom_type',
                                                                                   math.nan) is not math.nan else math.nan
    atm_stable = molecule.get('atm_stable', math.nan)
    mol_stable = molecule.get('mol_stable', math.nan)
    validity_rdkit = molecule.get('validity_rdkit', math.nan)
    uniqueness_rdkit = molecule.get('uniqueness_rdkit', math.nan)
    novelty_rdkit = molecule.get('novelty_rdkit', math.nan)
    positions = x[:long].detach().cpu()
    atom_type = atom_type[:long].detach().cpu()

    plot_data3d(positions, atom_type, dataset_info=dataset_info, save_path='./EvoResults/'  + str(type) + str(index) +
                                                                           '_as' + str(atm_stable) +
                                                                           '_ms' + str(mol_stable) +
                                                                           '_val' + str(validity_rdkit) +
                                                                           '_uni' + str(uniqueness_rdkit) +
                                                                           '_nov' + str(novelty_rdkit)
                                                                           +'.png',
                spheres_3d=True)




def find_subgraph_indices(adj_matrix, L):
    """ BFS 寻找大小为 L 的连通子图索引 """
    k = adj_matrix.shape[0]
    for start in range(k):
        visited = {start}
        queue = [start]
        while queue and len(visited) < L:
            current = queue.pop(0)
            neighbors = np.where(adj_matrix[current] != 0)[0].tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) == L: break
            if len(visited) == L: break
        if len(visited) == L:
            return sorted(list(visited))
    return None

def set_seed(seed):
    """设置全局随机种子"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # 某些操作可能需要确定性算法才能完全复现，但这可能会降低性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def save_random_states():
    """保存当前 random、numpy 和 torch 的随机状态"""
    states = {}
    states['random'] = random.getstate()
    states['numpy'] = np.random.get_state()
    states['torch_cpu'] = torch.get_rng_state()
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        states['torch_cuda'] = [torch.cuda.get_rng_state(d) for d in range(num_devices)]
    return states

def restore_random_states(states):
    """恢复之前保存的随机状态"""
    random.setstate(states['random'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch_cpu'])
    if torch.cuda.is_available() and 'torch_cuda' in states:
        num_devices = torch.cuda.device_count()
        for d in range(num_devices):
            torch.cuda.set_rng_state(states['torch_cuda'][d], d)
def Get_Multipattern(args, device, model_sample, dataset_info, prop_dist, min_n_nodes, max_n_nodes, preds, L, seed=None):
    """
    args: 参数配置
    L: int 或 list[int]。如果是 list，则返回多个片段（每个片段来自不同的分子）。
    seed: int, 随机种子。
    """
    max_pad = 29  # 原代码中的填充长度限制

    # 1. 设置随机种子
    if seed is not None:
        saved_states = save_random_states()
        set_seed(seed)  # 你的原有 set_seed 函数
    try:
        # 2. 规范化 L 参数
        # 如果 L 是单个整数或 None，将其转换为列表处理，最后再解包返回
        is_single_request = not isinstance(L, list)
        target_sizes = [L] if is_single_request else L

        results = []

        # 3. 针对每个目标大小进行独立的采样和裁剪
        for target_L in target_sizes:
            while True:
                # --- A. 采样初始分子 ---
                val = torch.randint(min_n_nodes, max_n_nodes + 1, (1,)).item()
                nodesxsample = torch.full((1,), val)

                # sample 返回值保持 [1, N, ...] 形状
                one_hot, charges, x, node_mask, z = sample(args, device, model_sample, dataset_info, prop_dist,
                                                           nodesxsample=nodesxsample)

                # 构造用于 analyze 的 molecule 字典 (输入形状不变)
                molecule = {
                    'one_hot': one_hot,
                    'x': x,
                    'node_mask': node_mask,
                    'z': z,
                    'T': 0,
                    'AddT': 0,
                    'long': int(node_mask.sum().item()),
                    'atom_type': torch.argmax(one_hot.squeeze(0), dim=1)
                }

                # --- B. 调用外部稳定性检查 ---
                validity_dict, rdkit_tuple = analyze.analyze_stability_for_molecules(molecule, dataset_info)

                # 校验不通过则重新开始循环 (继续采样新分子)
                if validity_dict['atm_stable'] != 1.0 or validity_dict['mol_stable'] != 1.0 or rdkit_tuple[0][0] != 1.0:
                    continue

                # 稳定性通过后，将内部存储的 Tensor 挤压掉 batch 维度以供后续处理
                molecule['one_hot'] = molecule['one_hot'].squeeze(0)
                molecule['x'] = molecule['x'].squeeze(0)
                molecule['node_mask'] = molecule['node_mask'].squeeze(0)
                molecule['z'] = molecule['z'].squeeze(0)

                # --- C. 如果不需要裁剪片段 (target_L is None) ---
                if target_L is None:
                    # 这种情况下没有 fragment 信息，填充 None
                    results.append((molecule, None, None, None))
                    break  # 跳出 while，处理下一个 target_size

                # --- D. 子图裁剪逻辑 ---
                # 准备邻接矩阵
                x_active = molecule['x'][:molecule['long']].cpu()
                decoded_types = [dataset_info['atom_decoder'][idx] for idx in
                                 molecule['atom_type'][:molecule['long']].cpu().numpy()]
                adj_matrix = get_bond_mat(x_active, decoded_types)

                # 寻找连通子图索引
                selected_indices = find_subgraph_indices(adj_matrix, target_L)
                if selected_indices is None:
                    continue  # 没找到指定大小的连通子图，丢弃当前分子，重试

                # 提取子图数据
                indices_tensor = torch.tensor(selected_indices, device=device)
                x_sub = molecule['x'][indices_tensor]
                at_sub = molecule['atom_type'][indices_tensor]
                oh_sub = F.one_hot(at_sub, num_classes=len(dataset_info['atom_decoder'])).float()
                z_sub = molecule['z'][indices_tensor]

                # --- E. 构造片段分子 molecule1 ---
                molecule1 = {
                    'long': target_L,
                    'T': 0,
                    'AddT': 0,
                    'x': F.pad(x_sub, (0, 0, 0, max_pad - target_L)),
                    'one_hot': F.pad(oh_sub, (0, 0, 0, max_pad - target_L)),
                    'atom_type': F.pad(at_sub, (0, max_pad - target_L)),
                }

                # 构造 node_mask [29, 1]
                mask_ones = torch.ones(target_L, 1, device=device)
                mask_zeros = torch.zeros(max_pad - target_L, 1, device=device)
                molecule1['node_mask'] = torch.cat([mask_ones, mask_zeros], dim=0)

                # --- F. VAE 编码与解码 ---
                atom_mask = torch.ones(1, target_L, device=device)
                edge_mask = (atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2))
                diag_mask = ~torch.eye(target_L, dtype=torch.bool, device=device).unsqueeze(0)
                edge_mask = (edge_mask * diag_mask).view(1, -1)

                # 中心化
                x_sub_centered = diffusion_utils.remove_mean_with_mask(x_sub.unsqueeze(0), atom_mask.unsqueeze(-1))

                if hasattr(model_sample, 'vae'):
                    # Encode
                    h_info = {'categorical': oh_sub.unsqueeze(0), 'integer': at_sub.unsqueeze(0).unsqueeze(-1)}
                    z_x_mu, _, z_h_mu, _ = model_sample.vae.encode(x_sub_centered, h_info, atom_mask.unsqueeze(-1), edge_mask)
                    z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)

                    # Decode
                    dec_x, dec_h = model_sample.vae.decode(z_xh_mean, atom_mask.unsqueeze(2), edge_mask, None)
                else:
                    z_xh_mean = z_sub

                # 存储 Z
                molecule1['z'] = F.pad(z_xh_mean, (0, 0, 0, max_pad - target_L)).squeeze(0).detach()

                # --- G. 保存结果并跳出当前 While ---
                # 结果元组: (原分子, 片段分子, 解码坐标, 解码原子类型)
                if hasattr(model_sample, 'vae'):
                    results.append(
                        (molecule, molecule1, dec_x.squeeze(0), torch.argmax(dec_h['categorical'].squeeze(0), dim=1)))
                else:
                    results.append((molecule, molecule1))
                break

                # 4. 返回结果
        if is_single_request:
            # 如果输入 L 是单个整数，返回单个元组（保持向后兼容）
            return results[0]
        else:
            # 如果输入 L 是列表，返回列表
            return results
    finally:
        # 无论成功或异常，只要之前保存了状态就恢复
        if saved_states is not None:
            restore_random_states(saved_states)

def Get_pattern(args, device, model_sample, dataset_info, prop_dist, min_n_nodes, max_n_nodes, preds, L):
    max_pad = 29  # 原代码中的填充长度限制

    while True:
        # 1. 采样初始分子
        val = torch.randint(min_n_nodes, max_n_nodes + 1, (1,)).item()
        nodesxsample = torch.full((1,), val)

        # sample 返回值保持 [1, N, ...] 形状
        one_hot, charges, x, node_mask, z = sample(args, device, model_sample, dataset_info, prop_dist,
                                                   nodesxsample=nodesxsample)

        # 构造用于 analyze 的 molecule 字典 (输入形状不变)
        molecule = {
            'one_hot': one_hot,
            'x': x,
            'node_mask': node_mask,
            'z': z,
            'T': 0,
            'AddT': 0,
            'long': int(node_mask.sum().item()),
            'atom_type': torch.argmax(one_hot.squeeze(0), dim=1)
        }

        # 2. 调用外部稳定性检查 (输入 molecule 包含 batch 维度)
        validity_dict, rdkit_tuple = analyze.analyze_stability_for_molecules(molecule, dataset_info)

        # 校验不通过则重新开始循环 (代替递归)
        if validity_dict['atm_stable'] != 1.0 or validity_dict['mol_stable'] != 1.0 or rdkit_tuple[0][0] != 1.0:
            continue

        # 稳定性通过后，将内部存储的 Tensor 挤压掉 batch 维度以供后续处理
        molecule['one_hot'] = molecule['one_hot'].squeeze(0)
        molecule['x'] = molecule['x'].squeeze(0)
        molecule['node_mask'] = molecule['node_mask'].squeeze(0)
        molecule['z'] = molecule['z'].squeeze(0)

        # 3. 如果不需要裁剪片段，直接返回
        if L is None:
            return molecule

        # 4. 子图裁剪逻辑
        # 准备邻接矩阵
        x_active = molecule['x'][:molecule['long']].cpu()
        decoded_types = [dataset_info['atom_decoder'][idx] for idx in
                         molecule['atom_type'][:molecule['long']].cpu().numpy()]
        adj_matrix = get_bond_mat(x_active, decoded_types)

        # 寻找连通子图索引
        selected_indices = find_subgraph_indices(adj_matrix, L)
        if selected_indices is None:
            continue  # 没找到大小为 L 的连通子图，重试

        # 提取子图数据
        indices_tensor = torch.tensor(selected_indices, device=device)
        x_sub = molecule['x'][indices_tensor]
        at_sub = molecule['atom_type'][indices_tensor]
        oh_sub = F.one_hot(at_sub, num_classes=len(dataset_info['atom_decoder'])).float()

        # 5. 构造片段分子 molecule1
        molecule1 = {
            'long': L,
            'T': 0,
            'AddT': 0,
            'x': F.pad(x_sub, (0, 0, 0, max_pad - L)),
            'one_hot': F.pad(oh_sub, (0, 0, 0, max_pad - L)),
            'atom_type': F.pad(at_sub, (0, max_pad - L)),
        }

        # 构造 node_mask [29, 1]
        mask_ones = torch.ones(L, 1, device=device)
        mask_zeros = torch.zeros(max_pad - L, 1, device=device)
        molecule1['node_mask'] = torch.cat([mask_ones, mask_zeros], dim=0)

        # 6. VAE 编码与解码
        atom_mask = torch.ones(1, L, device=device)
        edge_mask = (atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2))
        diag_mask = ~torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask = (edge_mask * diag_mask).view(1, -1)

        # 中心化
        x_sub_centered = diffusion_utils.remove_mean_with_mask(x_sub.unsqueeze(0), atom_mask.unsqueeze(-1))

        # Encode
        h_info = {'categorical': oh_sub.unsqueeze(0), 'integer': at_sub.unsqueeze(0).unsqueeze(-1)}
        z_x_mu, _, z_h_mu, _ = model_sample.vae.encode(x_sub_centered, h_info, atom_mask.unsqueeze(-1), edge_mask)
        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)

        # Decode
        dec_x, dec_h = model_sample.vae.decode(z_xh_mean, atom_mask.unsqueeze(2), edge_mask, None)

        # 存储并返回
        molecule1['z'] = F.pad(z_xh_mean, (0, 0, 0, max_pad - L)).squeeze(0).detach()

        return molecule, molecule1, dec_x.squeeze(0), torch.argmax(dec_h['categorical'].squeeze(0), dim=1)




def InitPop(NPops, nodes_dist, args, device, model_sample, dataset_info, prop_dist,
            isequal, min_n_nodes, preds, max_n_nodes, evalobj, context=None):
    """
    初始化分子种群，包含原子序数字段
    """
    # 0. 准备原子序数映射表 (H=1, C=6, N=7, O=8, F=9, etc.)
    symbol_to_z = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Al': 13, 'Si': 14, 'P': 15,
                   'S': 16, 'Cl': 17, 'As': 33, 'Br': 35, 'I': 53, 'Hg': 80, 'Bi': 83}
    z_list = [symbol_to_z[s] for s in dataset_info['atom_decoder']]
    z_map = torch.tensor(z_list, device=device, dtype=torch.long)

    # 1. 确定每个分子的原子数量 (向量化处理)
    if isequal:
        val = random.randint(min_n_nodes, max_n_nodes)
        nodesxsample = torch.full((NPops,), val, dtype=torch.long)
    else:
        nodesxsample = nodes_dist.sample(NPops, min_nodes=min_n_nodes, max_nodes=max_n_nodes).long()

    # 2. 调用扩散模型采样
    one_hot, charges, x, node_mask, z = sample_max_n(
        max_n_nodes, args, device, model_sample, dataset_info, prop_dist,
        nodesxsample=nodesxsample, context=context
    )

    # 3. 预计算所有分子的属性 (Batch 化)
    atom_types_batch = one_hot.argmax(dim=-1) # [NPops, max_nodes]
    # 将索引映射为原子序数: [NPops, max_nodes]
    atomic_numbers_batch = z_map[atom_types_batch]
    lengths_batch = node_mask.sum(dim=1).long().cpu() # [NPops]

    # 4. 构建分子种群列表
    molecules = [
        {
            'one_hot': one_hot[i],
            'x': x[i],
            'node_mask': node_mask[i],
            'z': z[i],
            'T': 0,
            'AddT': 0,
            'long': lengths_batch[i].item(),
            'atom_type': atom_types_batch[i],
            'atomic_numbers': atomic_numbers_batch[i], # 新增字段
            'EvaluatedVina': False,
            'EvaluatedPB': False,
            'EvaluatedSC': False
        }
        for i in range(NPops)
    ]

    # 5. 评价
    if evalobj:
        molecules = EvalPop_Con(molecules, dataset_info)
        if preds is not None:
            molecules = EvalPop_Obj(molecules, device, preds, max_n_nodes)

    return molecules

def calculate_radius_of_gyration(mol, include_hydrogens=False):

    # 检查是否存在3D构象
    if mol.GetNumConformers() == 0:
        raise ValueError("分子缺少3D坐标，请先用 AllChem.EmbedMolecule() 生成构象")

    conf = mol.GetConformer()
    atoms = []
    positions = []
    masses = []

    # 遍历原子并过滤氢原子
    for atom in mol.GetAtoms():
        if not include_hydrogens and atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom)
        positions.append([pos.x, pos.y, pos.z])
        masses.append(atom.GetMass())

    if len(atoms) == 0:
        raise ValueError("没有可用于计算的原子（可能因过滤氢原子导致）")

    # 转换为numpy数组加速计算
    positions = np.array(positions)
    masses = np.array(masses)

    # 计算质心（质量加权平均）
    centroid = np.sum(positions * masses[:, np.newaxis], axis=0) / masses.sum()

    # 计算回转半径
    squared_dist = np.sum((positions - centroid) ** 2, axis=1)
    rg = np.sqrt(np.dot(masses, squared_dist) / masses.sum())

    return round(rg, 3)






# def sdf_to_pdbqt(sdf_file, pdbqt_outfile, conda_bin_dir=""):
#     obabel_exec_path = os.path.join(conda_bin_dir, "obabel")
#     os.environ['BABEL_LIBDIR'] = '/data/srq/conda/envs/DEMO310/lib/openbabel/3.1.0/'
#     os.popen(f'{obabel_exec_path} {sdf_file} -O {pdbqt_outfile}').read()
import subprocess


def shift_mol_in_Pop(Pop,cx,cy,cz,device,shiftfromcopy = False):
    cent = torch.tensor([cx, cy, cz]).to(device)
    if shiftfromcopy:
        for mol in Pop:
            #mol['x'] = remove_mean_with_mask(mol['x'],mol['node_mask'])
            for i in range(mol['long']):
                mol['x'][i] = mol['x_ori'][i] + cent
    else:
        for mol in Pop:
            #mol['x'] = remove_mean_with_mask(mol['x'],mol['node_mask'])
            for i in range(mol['long']):
                mol['x'][i] = mol['x'][i] + cent




def evaluate_PB_pop(Pop, prefix,pocket_pdb, protein_dir,vinaname='vina',pbname = 'PBCon'):
    from openbabel import pybel
    os.environ['BABEL_LIBDIR'] = 'D:\\anaconda3\\envs\\cgm310win\\Lib\\openbabel'
    #os.environ['BABEL_LIBDIR'] = '/your_conda_env_dir/lib/openbabel/3.1.0/' #if linux
    NPops = len(Pop)
    for i in range(NPops):
        if Pop[i][vinaname] < 0 and Pop[i]['EvaluatedPB'] == False:
            ligand = protein_dir + '/' + prefix + str(i)
            mol = next(pybel.readfile("pdbqt", ligand + '_out.pdbqt'))
            mol.addh()
            mol.write("sdf", ligand + "_H_bestvina.sdf", overwrite=True)
            buster = PoseBusters(config="dock")
            df_results = buster.bust(
                mol_pred=ligand + "_H_bestvina.sdf",
                mol_true=None,
                mol_cond=pocket_pdb,
                full_report=False
            )
            unpass_rate = 1-(df_results.select_dtypes(include="bool").sum(axis=1).iloc[0] / df_results.shape[1])
            Pop[i][pbname] = unpass_rate
            Pop[i]['EvaluatedPB'] = True
        else:
            Pop[i][pbname] = 1


import warnings
from rdkit import RDLogger
import contextlib
@contextlib.contextmanager
def stderr_redirected(to=os.devnull):
    """
    将 stderr 文件描述符重定向到指定文件（默认为 os.devnull）。
    此操作影响整个进程，因此会临时隐藏所有线程的输出。
    """
    fd = sys.stderr.fileno()
    # 保存原始 stderr 文件描述符
    saved_fd = os.dup(fd)
    # 打开目标文件
    with open(to, 'wb') as f:
        os.dup2(f.fileno(), fd)
    try:
        yield
    finally:
        # 恢复原始 stderr
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


def evaluate_PB_pop_MT(Pop, prefix, pocket_pdb, protein_dir, vinaname='vina', pbname='PBCon', n_worker=9):
    """
    评估种群中分子的 PoseBusters 通过率（多线程版）。
    """
    num_pop = len(Pop)

    # 定义单个分子的处理函数
    def process_one(i, mol):
        # 只处理 vina 分数 < 0 且未评估 PB 的分子
        if mol[vinaname] < 0 and mol.get('EvaluatedPB', False) is False:
            try:
                from openbabel import pybel
                # 构造文件名（基于索引保证唯一）
                ligand_base = f"{protein_dir}/{prefix}{i}"
                pdbqt_file = f"{ligand_base}_out.pdbqt"
                sdf_file = f"{ligand_base}_H_bestvina.sdf"

                # 读取 pdbqt 并加氢，保存为带氢的 sdf
                mol_pybel = next(pybel.readfile("pdbqt", pdbqt_file))
                mol_pybel.addh()
                mol_pybel.write("sdf", sdf_file, overwrite=True)

                with stderr_redirected():
                    buster = PoseBusters(config="dock")
                    df_results = buster.bust(
                        mol_pred=sdf_file,
                        mol_true=None,
                        mol_cond=pocket_pdb,
                        full_report=False
                    )

                # 计算未通过率
                bool_cols = df_results.select_dtypes(include="bool")
                unpass_rate = 1 - (bool_cols.sum(axis=1).iloc[0] / df_results.shape[1])

                # 更新分子属性
                mol[pbname] = unpass_rate
                mol['EvaluatedPB'] = True

            except Exception as e:
                # 如果出现错误，记录并设置为默认值 1
                print(f"Error processing molecule {i}: {e}")
                mol[pbname] = 1
                mol['EvaluatedPB'] = True  # 避免重复尝试失败的任务
        else:
            # 不满足条件时直接赋值 1
            mol[pbname] = 1

    # 使用线程池执行并行任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_worker) as executor:
        # 提交所有任务，建立 future 到索引的映射
        future_to_index = {executor.submit(process_one, i, mol): i for i, mol in enumerate(Pop)}

        # 用 tqdm 在主线程更新进度条
        with tqdm(total=num_pop, desc=f"PBCon({prefix})", leave=False) as pbar:
            for future in concurrent.futures.as_completed(future_to_index):
                # 捕获可能的异常（已在 process_one 内部处理，此处仅用于更新进度条）
                try:
                    future.result()  # 如果任务抛出未捕获的异常，会在这里抛出
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"Unhandled error for molecule {idx}: {e}")
                finally:
                    pbar.update(1)

    return Pop


import os
import numpy as np
from pathlib import Path
from rdkit import Chem


def prepare_pocket_info(pocket_index, protein_dir, conda_bin_dir="", num_repeats=3):
    """
    打包预处理逻辑，并自动多次计算原始配体的 Baseline 分数取均值
    num_repeats: 计算基准分数的重复次数（平滑 Vina 搜索的随机性）
    """
    # 【修复】：加上 sorted() 强制按文件名字母顺序排列
    all_pockets = sorted(list(Path('pdbqt/CD').glob('*.pdbqt')))
    random.seed(42)
    test_pockets = random.sample(all_pockets, 10)

    pocket_name = test_pockets[pocket_index].stem
    protein_name = pocket_name.split('-')[0]

    # 确保输出目录存在
    out_prot_dir = os.path.join(protein_dir, protein_name)
    os.makedirs(out_prot_dir, exist_ok=True)

    # 解析参考配体
    ref_path = list(Path('CDtest').glob(pocket_name + '*.sdf'))[0]
    # 这里开启 sanitize=True 是为了后续算 QED 和 SA 不报错
    ref_mol = next(Chem.SDMolSupplier(str(ref_path), sanitize=True))

    # 获取几何信息
    cx, cy, cz = ref_mol.GetConformer().GetPositions().mean(0)
    n_atoms = ref_mol.GetNumAtoms()
    # 假设 calculate_radius_of_gyration 在外部已定义
    r_g = calculate_radius_of_gyration(ref_mol) * 3

    # --- 2. 计算 Vina Score, QED, SA (多次计算取均值) ---
    # 先用 obabel 将原始 sdf 转为 pdbqt
    ref_pdbqt_path = os.path.join(out_prot_dir, f"{pocket_name}_ref.pdbqt")
    sdf_to_pdbqt(ref_path, ref_pdbqt_path, conda_bin_dir)

    print(f"    [Baseline] Calculating reference scores for {pocket_name} ({num_repeats} repeats)...")

    vinas, qeds, sas = [], [], []

    for i in range(num_repeats):
        try:
            # 注意：让原生配体也在相同的 Box 下重对接一次，保证比较的公平性
            r_vina, r_qed, r_sa = calculate_qvina2_score_lite(
                receptor_file=f"pdbqt/CD/{pocket_name}.pdbqt",
                sdf_file=str(ref_path),
                ligand_pdbqt=ref_pdbqt_path,
                out_dir=out_prot_dir,
                size=r_g,
                conda_bin_dir=conda_bin_dir,
                exhaustiveness=16  # 保持和主循环一致
            )
            vinas.append(r_vina)
            qeds.append(r_qed)
            sas.append(r_sa)
        except Exception as e:
            print(f"      Warning: Failed to calculate reference score on repeat {i + 1}: {e}")

    # 计算平均值（如果全部失败则给默认兜底值）
    ref_vina = np.mean(vinas) if vinas else 0.0
    ref_qed = np.mean(qeds) if qeds else 0.0
    ref_sa = np.mean(sas) if sas else 10.0  # SA 越小越好，失败给惩罚值 10.0

    return {
        "pocket_name": pocket_name,
        "protein_name": protein_name,
        "protein_dir": out_prot_dir,
        "pocket_pdb": f'./CDtest/{pocket_name}.pdb',
        "center": (cx, cy, cz),
        "n_atoms": n_atoms,
        "r_g": r_g,
        "max_n_nodes": int(n_atoms * 1.3),
        "min_n_nodes": int(n_atoms * 0.7),
        # === 记录 Baseline 指标 ===
        "ref_vina": ref_vina,
        "ref_qed": -ref_qed,  # 取负适配最小化逻辑
        "ref_sa": ref_sa,
        "vina": ref_vina,
        "qed": -ref_qed,  # 同上
        "sa": ref_sa,
        "summary_dir": out_prot_dir + '_summary',
    }


def get_pop_stats(pop):
    """
    统计种群的 Vina 性能指标
    """
    vina_scores = [p['fitness'] for p in pop if p['fitness'] < 0]
    feasible_count = sum(1 for p in pop if p['fitness'] < 0)

    stats = {
        "mean_vina": np.mean(vina_scores) if vina_scores else 0,
        "std_vina": np.std(vina_scores, ddof=1) if len(vina_scores) > 1 else 0,
        "best_vina": min(vina_scores) if vina_scores else 0,
        "feasible_rate": feasible_count / len(pop),
        "success_count": feasible_count
    }
    return stats

def cal_fitness_ring(Pop, epsilon):
    """
    评估并更新每个个体的 ring 相关惩罚（写回 Pop[i]['fitness']）。
    - 保持原签名，不改外部调用方式。
    - 强制必须包含至少一个环（ringcnt >= 1），否则视为严重违规。
    - allowed rings 根据 MW 自适应：allowed = min(1 + int(mw//100), max_cap)
    - 3-member ring 与 macrocycle 被视为硬违规（可改为软惩罚）
    - epsilon 为容忍值（总体惩罚在减去 epsilon 后下限为 0）
    """
    NPops = len(Pop)
    # 参数（可按需调整）
    max_cap = 6  # allowed 上限，防止允许值过大

    for i in range(NPops):
        entry = Pop[i]

        # 安全读取
        ringcnt = int(entry.get('ringcnt', 0))
        r3cnt = int(entry.get('r3cnt', 0))
        macro = int(entry.get('macrocycle_present', 0))
        r4cnt = int(entry.get('r4cnt', 0))
        fused = int(entry.get('fused_ring_count', 0))
        spiro = int(entry.get('spiro_count', 0))
        bridge = int(entry.get('bridgehead_count', 0))
        mw = entry.get('mw', None)
        heavy = entry.get('heavy_atoms', None)

        # 计算 allowed（基于 MW 优先，否则用 heavy_atoms，否则默认 4）
        if mw is not None:
            try:
                allowed = 1 + int(float(mw) // 100)
            except Exception:
                allowed = 4
        elif heavy is not None:
            try:
                allowed = 1 + int(int(heavy) // 10)
            except Exception:
                allowed = 4
        else:
            allowed = 4
        allowed = min(allowed, max_cap)

        # 逐项计算 violation（非负），均为原始计数/超出量（便于后续归一化/加权）
        violations = {}
        # 必须包含至少 1 个环：missing_ring = 1 if none, else 0
        violations['missing_ring'] = 1.0 if ringcnt == 0 else 0.0
        # 超出允许环数的数量（raw）
        violations['ring_excess'] = float(max(0, ringcnt - allowed))
        # 3-member rings：直接使用计数（通常 0/1）
        violations['r3_count'] = float(r3cnt)
        # macrocycle：直接使用计数/标志（通常 0/1）
        violations['macrocycle'] = float(macro)
        # 4-member rings：超出允许（允许 1 个）
        violations['r4_excess'] = float(max(0, r4cnt - 1))
        # fused rings：超出允许（允许 2）
        violations['fused_excess'] = float(max(0, fused - 2))
        # spiro / bridgehead：超出允许（允许 1）
        violations['spiro_excess'] = float(max(0, spiro - 1))
        violations['bridgehead_excess'] = float(max(0, bridge - 1))
        # 保留环信息便于诊断
        # violations['ringcnt'] = float(ringcnt)
        entry['allowed_rings'] = float(allowed)

        # 汇总违反量（原始和 epsilon 调整）
        total_violation = sum(violations[k] for k in violations)
        total_violation_eps = max(total_violation - float(epsilon), 0.0)

        # 写回 violations
        entry['cons_violations'] = violations
        entry['total_violation'] = float(total_violation)
        entry['total_violation_eps'] = float(total_violation_eps)

        # 按你原始逻辑更新 fitness（保持惯例：当 fitness < 0 时替换；否则累加）
        # （注意：这里不改原逻辑，仅更新栈内值）
        if total_violation_eps != 0 and entry.get('fitness', 0.0) < 0:
            entry['fitness'] = total_violation_eps
        elif total_violation_eps != 0 and entry.get('fitness', 0.0) >= 0:
            entry['fitness'] = entry.get('fitness', 0.0) + total_violation_eps
        # 若 con == 0 则不改 fitness（与你原函数行为一致）


@torch.no_grad()
def evaluate_vina_pop(Pop, prefix, pocket_name, conda_bin_dir, protein_dir, atom_decoder, dataset_info, protein_name,
                      oPop=None, draw=False, r_g=None):
    """
    评估种群中所有分子的 Vina 对接分数。
    """
    num_pop = len(Pop)
    # 使用 tqdm 代替手动打印进度条
    pbar = tqdm(enumerate(Pop), total=num_pop, desc=f"Vina({prefix})", leave=False)

    for i, mol in pbar:

        if mol['EvaluatedVina'] == False:

            # 2. 构建路径
            base_name = os.path.join(protein_dir, f"{prefix}{i}")
            sdf_path = f"{base_name}.sdf"
            pdbqt_path = f"{base_name}.pdbqt"
            receptor_pdbqt = os.path.join('pdbqt', 'CD', f"{pocket_name}.pdbqt")

            # 3. 转换为 SDF 并执行对接流程
            # 假设 tensor_to_sdf 成功返回 True
            success_sdf = tensor_to_sdf(mol, base_name, atom_decoder)

            if success_sdf:
                # SDF -> PDBQT
                sdf_to_pdbqt(sdf_path, pdbqt_path, conda_bin_dir)

                # 注意：ind=mol 会直接将 QED, LogP, Fingerprint 等写入 mol 字典
                vina, qed, sa = calculate_qvina2_score_lite(
                    receptor_file=receptor_pdbqt,
                    sdf_file=sdf_path,
                    ligand_pdbqt=pdbqt_path,
                    out_dir=protein_dir,
                    return_rdmol=False,
                    conda_bin_dir=conda_bin_dir,
                    r_g=r_g,
                    ind=mol
                )

                # 4. 更新分子字典属性
                mol['vina'] = torch.tensor(vina)
                mol['qed'] = torch.tensor(-qed)
                mol['sa'] = torch.tensor(sa)
                mol['EvaluatedVina'] = True
                # 5. 异常可视化：如果对接失败 (score=0) 且开启 draw，保存原构象 3D 图用于分析
                if vina == 0 and draw and oPop is not None:
                    mol['EvaluatedVina'] = False
                    save_dir_prefix = f"ProteinPocket/{protein_name}/{prefix}_"
                    save3D(oPop[i], dataset_info, save_dir_prefix, i)

        # 更新 tqdm 右侧显示的实时分值（可选）
        # pbar.set_postfix({'vina': f"{mol['vina']:.2f} | 'qed' :{mol['qed']:.2f} | 'sa': {mol['sa']:.2f}"})

    return Pop

import concurrent.futures
from tqdm import tqdm

@torch.no_grad()
def evaluate_vina_pop_MT(Pop, prefix, pocket_name, conda_bin_dir, protein_dir, atom_decoder, dataset_info, protein_name,
                      oPop=None, draw=False, r_g=None, n_workers=18):
    """
    评估种群中所有分子的 Vina 对接分数（多线程版）。
    """
    num_pop = len(Pop)

    # 定义单个分子的处理函数
    def process_one(i, mol):
        if mol['EvaluatedVina']:
            return  # 已评估则跳过

        # 构建唯一文件路径（基于索引，避免冲突）
        base_name = os.path.join(protein_dir, f"{prefix}{i}")
        sdf_path = f"{base_name}.sdf"
        pdbqt_path = f"{base_name}.pdbqt"
        receptor_pdbqt = os.path.join('pdbqt', 'CD', f"{pocket_name}.pdbqt")

        # 将分子张量转换为 SDF 文件
        success_sdf = tensor_to_sdf(mol, base_name, atom_decoder)

        if success_sdf:
            # SDF -> PDBQT 格式转换
            sdf_to_pdbqt(sdf_path, pdbqt_path, conda_bin_dir)

            # 运行 Vina 对接并获取分数（ind=mol 会直接更新 mol 字典）
            vina, qed, sa = calculate_qvina2_score_lite_single(
                receptor_file=receptor_pdbqt,
                sdf_file=sdf_path,
                ligand_pdbqt=pdbqt_path,
                out_dir=protein_dir,
                return_rdmol=False,
                conda_bin_dir=conda_bin_dir,
                r_g=r_g,
                ind=mol
            )

            # 更新分子属性
            mol['vina'] = torch.tensor(vina)
            mol['qed'] = torch.tensor(-qed)
            mol['sa'] = torch.tensor(sa)
            mol['EvaluatedVina'] = True

            # 异常情况可视化（对接分数为0时）
            if vina == 0 and draw and oPop is not None:
                mol['EvaluatedVina'] = False  # 标记为未评估（可能重试）
                save_dir_prefix = f"ProteinPocket/{protein_name}/{prefix}_"
                save3D(oPop[i], dataset_info, save_dir_prefix, i)
        # 若 success_sdf 为 False，则未更新 EvaluatedVina，保持原样

    # 使用线程池执行并行任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        # 提交所有任务，并建立 future 到索引的映射
        future_to_index = {executor.submit(process_one, i, mol): i for i, mol in enumerate(Pop)}

        # 使用 tqdm 在主线程中更新进度条
        with tqdm(total=num_pop, desc=f"Vina({prefix})", leave=False) as pbar:
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    future.result()  # 获取结果（如果任务抛出异常会在这里触发）
                except Exception as e:
                    idx = future_to_index[future]
                    print(f"Error processing molecule {idx}: {e}")
                finally:
                    pbar.update(1)

    return Pop


import os
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import QED
import sys
from rdkit import RDConfig

# 确保 SA_Score 模块可以被导入
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import os
import sys
import shutil


def get_qvina_path():
    """
    全平台自适应寻找 qvina 可执行文件路径
    """
    env_root = sys.prefix
    qvina_exec_path = None

    # 定义可能的可执行文件名称（涵盖 Windows 的 .exe 和 Linux 的无后缀名，以及可能的版本简写）
    exec_names = ['qvina2.1.exe', 'qvina2.exe', 'qvina.exe', 'qvina2.1', 'qvina2', 'qvina']

    potential_paths = []

    # 1. 根据系统拼装所有可能的物理路径
    if os.name == 'nt':  # Windows
        for name in exec_names:
            potential_paths.extend([
                os.path.join(env_root, "Library", "bin", name),  # Conda C/C++ 库默认路径
                os.path.join(env_root, "Scripts", name),  # Conda 脚本路径
                os.path.join(env_root, "bin", name),  # 某些包的不规范存放路径
                os.path.join(os.getcwd(), name)  # 当前工作目录 (防手动拷贝)
            ])
    else:  # Linux / Mac
        for name in exec_names:
            potential_paths.extend([
                os.path.join(env_root, "bin", name),
                os.path.join(os.getcwd(), name)
            ])

    # 2. 遍历物理路径进行精确打击
    for p in potential_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            qvina_exec_path = p
            break

    # 3. 如果物理路径没搜到，利用 shutil.which 在系统的全局环境变量 PATH 里扫底
    if not qvina_exec_path:
        for name in exec_names:
            path = shutil.which(name)
            if path:
                qvina_exec_path = path
                break

    # 4. 终极报错提示
    if not qvina_exec_path:
        raise FileNotFoundError(
            f"Error: Could not find any Quick Vina executable (e.g., qvina2.1, qvina2.1.exe) "
            f"in your conda environment ({env_root}) or system PATH.\n"
            f"Please ensure it is installed correctly."
        )

    return qvina_exec_path

def calculate_qvina2_score_lite_single(receptor_file, sdf_file, ligand_pdbqt, out_dir, size=20,
                                exhaustiveness=16, return_rdmol=False, conda_bin_dir="", r_g=None, ind=None):
    """
    轻量版评价函数：仅计算 Vina Score, QED, 和 SA Score。
    【重要优化】：限制 Vina 为单核执行，配合外层 Python 多线程实现最强并发。
    """
    import sascorer

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)

    # 1. 准备受体 PDBQT
    if receptor_file.suffix == '.pdb':
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        if not receptor_pdbqt_file.exists():  # 避免多线程重复生成受体
            prepare_receptor_script_path = os.path.join(conda_bin_dir, "prepare_receptor4.py")
            subprocess.run([prepare_receptor_script_path, '-r', str(receptor_file), '-O', str(receptor_pdbqt_file)],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False
                           )
    else:
        receptor_pdbqt_file = receptor_file

    scores = 0.0
    qed_val = 0.0
    sa_val = 10.0  # 惩罚分
    rdmols = []

    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    for i, mol in enumerate(suppl):
        ligand_name = f'{sdf_file.stem}_{i}'
        out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        # 2. 检查 3D 构象并获取中心点
        if mol is None: continue

        try:
            cx, cy, cz = mol.GetConformer().GetPositions().mean(0)
        except Exception:
            cx, cy, cz = 0.0, 0.0, 0.0  # 异常兜底

        # 3. 计算 QED 和 SA Score
        try:
            qed_val = QED.qed(mol)
        except:
            qed_val = 0.0

        try:
            sa_val = sascorer.calculateScore(mol)
        except:
            sa_val = 10.0

        # 4. 运行 QuickVina 2
        qvina_exec_path = get_qvina_path()
        box_size = r_g if r_g is not None else size

        cmd = [
            qvina_exec_path,
            '--receptor', str(receptor_pdbqt_file),
            '--ligand', str(ligand_pdbqt),
            '--center_x', f'{cx:.4f}',
            '--center_y', f'{cy:.4f}',
            '--center_z', f'{cz:.4f}',
            '--size_x', str(box_size),
            '--size_y', str(box_size),
            '--size_z', str(box_size),
            '--exhaustiveness', str(exhaustiveness),
            '--cpu', '1'  # <========= 【核心关键】：强制单核！防止多线程灾难！
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 缩小超时时间，防止畸形分子卡死 Vina
                check=False
            )
            out = result.stdout
        except subprocess.TimeoutExpired:
            out = ""
        except Exception:
            out = ""

        # 解析 Vina 输出
        out_split = out.splitlines()
        try:
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores = float(best_line[1])
        except (ValueError, IndexError, AssertionError):
            scores = 0.0

        # 5. OpenBabel 格式转换 (pdbqt -> sdf)
        # 注意：如果在 EA 循环中不需要返回 3D rdmol 对象，这段转换非常消耗 IO，建议跳过！
        if return_rdmol:
            out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
            obabel_exec_path = os.path.join(conda_bin_dir, "obabel")
            if out_pdbqt_file.exists():
                subprocess.run([obabel_exec_path, str(out_pdbqt_file), '-O', str(out_sdf_file)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    rdmol = Chem.SDMolSupplier(str(out_sdf_file))[0]
                    if rdmol: rdmols.append(rdmol)
                except:
                    pass

    if return_rdmol:
        return scores, rdmols
    else:
        return scores, qed_val, sa_val

def calculate_qvina2_score_lite(receptor_file, sdf_file, ligand_pdbqt, out_dir, size=20,
                                exhaustiveness=16, return_rdmol=False, conda_bin_dir="", r_g=None, ind=None):
    """
    轻量版评价函数：仅计算 Vina Score, QED, 和 SA Score。
    """
    import sascorer

    receptor_file = Path(receptor_file)
    sdf_file = Path(sdf_file)

    # 1. 准备受体 PDBQT (如果是 PDB 格式则转换)
    if receptor_file.suffix == '.pdb':
        receptor_pdbqt_file = Path(out_dir, receptor_file.stem + '.pdbqt')
        prepare_receptor_script_path = os.path.join(conda_bin_dir, "prepare_receptor4.py")
        # os.popen(f'{prepare_receptor_script_path} -r {receptor_file} -O {receptor_pdbqt_file}')
        result = subprocess.run(
            [prepare_receptor_script_path, '-r', str(receptor_file), '-O', str(receptor_pdbqt_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
    else:
        receptor_pdbqt_file = receptor_file

    scores = 0.0
    qed_val = 0.0
    sa_val = 10.0  # SA 分数越低越好(1-10)，如果计算失败，给最差的惩罚分 10.0
    rdmols = []

    suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False)
    for i, mol in enumerate(suppl):
        ligand_name = f'{sdf_file.stem}_{i}'
        out_sdf_file = Path(out_dir, ligand_name + '_out.sdf')

        # 2. 检查 3D 构象并获取中心点
        cx, cy, cz = mol.GetConformer().GetPositions().mean(0)

        # 3. 计算 QED 和 SA Score
        if mol is not None:
            try:
                qed_val = QED.qed(mol)
            except Exception:
                qed_val = 0.0

            try:
                sa_val = sascorer.calculateScore(mol)
            except Exception:
                sa_val = 10.0

        qvina_exec_path = get_qvina_path()
        box_size = r_g if r_g is not None else size

        cmd = [
            qvina_exec_path,
            '--receptor', str(receptor_pdbqt_file),
            '--ligand', ligand_pdbqt,
            '--center_x', f'{cx:.4f}',
            '--center_y', f'{cy:.4f}',
            '--center_z', f'{cz:.4f}',
            '--size_x', str(box_size),
            '--size_y', str(box_size),
            '--size_z', str(box_size),
            '--exhaustiveness', str(exhaustiveness)
        ]

        # 执行命令，捕获 stdout，丢弃 stderr
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,  # 捕获 stdout 和 stderr（Python 3.7+）
                text=True,  # 以文本形式返回输出
                timeout=300,  # 设置超时（秒），防止卡死
                check=False  # 不主动抛出异常（返回码非零时不报错）
            )
            out = result.stdout  # 仅使用 stdout
            # 如果需要，也可以检查 result.returncode 来判断执行是否成功
        except subprocess.TimeoutExpired:
            out = ""  # 超时时输出为空
        except Exception as e:
            out = ""  # 其他异常也置空

        # 原有的输出解析逻辑保持不变
        out_split = out.splitlines()
        try:
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            scores = float(best_line[1])
        except (ValueError, IndexError, AssertionError):
            scores = 0

        # 5. OpenBabel 格式转换 (pdbqt -> sdf)
        out_pdbqt_file = Path(out_dir, ligand_name + '_out.pdbqt')
        obabel_exec_path = os.path.join(conda_bin_dir, "obabel")
        if out_pdbqt_file.exists():
            os.popen(f'{obabel_exec_path} {out_pdbqt_file} -O {out_sdf_file}').read()

        if return_rdmol:
            try:
                rdmol = Chem.SDMolSupplier(str(out_sdf_file))
                rdmols.append(rdmol)
            except:
                pass

    if return_rdmol:
        return scores, rdmols
    else:
        # 现在的返回值变了，解包时请注意接收三个变量
        return scores, qed_val, sa_val

@torch.no_grad()
def Get_Fitness_multi_dis_MAE_normalize(Pop, dataset_info, device, preds, n_nodes, ObjName, patt, tolrance, ObjValue):
    # 1. 基础评估：计算 pattern 匹配、结构约束和目标属性值
    Pop = EvalPop_Patt(Pop, patt, dataset_info, tolrance)
    Pop = EvalPop_Con(Pop, dataset_info)
    Pop = EvalPop_Obj(Pop, device, preds, n_nodes)

    num_pop = len(Pop)
    if num_pop == 0:
        return Pop

    eps = 1e-12
    num_objs = len(ObjName)

    # 2. 统计各目标在当前种群中的范围，并预计算 MAE
    pop_obj_data = {name: [] for name in ObjName}
    for i in range(num_pop):
        mae_res = []
        for name in ObjName:
            val = float(Pop[i][name].item())
            pop_obj_data[name].append(val)
            # 记录原始 MAE (未归一化)
            mae_res.append(abs(val - ObjValue[name]))
        Pop[i]['MAE'] = mae_res

    # 计算每个维度的归一化参数 (min-max)
    bounds = {}
    for name in ObjName:
        vals = np.array(pop_obj_data[name])
        v_min, v_max = vals.min(), vals.max()
        # 如果当前种群在该维度值全部相同，则不进行缩放（分母设为1）
        denom = (v_max - v_min) if (v_max - v_min) > eps else 1.0
        bounds[name] = (v_min, denom)

    # 3. 计算每个个体到目标点 (ObjValue) 的归一化距离
    worst_dis = 0.0
    for i in range(num_pop):
        norm_coords = []
        for name in ObjName:
            v_min, denom = bounds[name]
            # 归一化当前个体值
            val_norm = (Pop[i][name].item() - v_min) / denom
            # 归一化目标值 (Target)
            target_norm = (ObjValue[name] - v_min) / denom
            norm_coords.append(val_norm - target_norm)

        # 计算欧氏距离 (对于单目标，np.linalg.norm 等价于 abs())
        dis_i = float(np.linalg.norm(norm_coords))
        Pop[i]['dis'] = dis_i

        if dis_i > worst_dis:
            worst_dis = dis_i

    # 4. 结合约束设置 Fitness (使用惩罚函数法)
    for i in range(num_pop):
        # 综合考虑属性约束和物理结构约束
        struct_con_data = Pop[i].get('structure_Con', [0])
        if isinstance(struct_con_data, list):
            struct_con_sum = sum(struct_con_data)
        else:
            struct_con_sum = struct_con_data

        struct_con_sum = max(0, struct_con_sum)

        chem_con = max(0, Pop[i].get('Constraint', 0))

        total_con = struct_con_sum + chem_con

        if total_con <= 0:
            # 可行解：fitness 就是距离目标点的距离
            Pop[i]['fitness'] = Pop[i]['dis']
        else:
            # 不可行解：施加惩罚，确保其 fitness 差于任何可行解
            # fitness = 归一化距离 + 种群中最差距离惩罚
            Pop[i]['fitness'] = Pop[i]['dis'] + worst_dis

    return Pop

def EnvironmentalSelectionCon(Pop, N, ObjName):
    fitness = []

    for i in range(len(Pop)):
        fitness.append(Pop[i]['fitness'])
    fitness = np.array(fitness)

    Obj = []
    for j in range(len(Pop)):
        res = []
        for obj in ObjName:
            res.append(Pop[j][obj].item() if isinstance(Pop[j][obj], torch.Tensor) else Pop[j][obj])
        Obj.append(deepcopy(res))
    Obj = np.array(Obj)

    # Environmental selection
    next_gen = fitness < 1
    if sum(next_gen) < N:
        rank = np.argsort(fitness)
        next_gen[rank[:N]] = True
    elif sum(next_gen) > N:
        del_indices = truncation(Obj[next_gen], sum(next_gen) - N)
        temp = np.where(next_gen)[0]
        next_gen[temp[del_indices]] = False

    # Population for next generation
    population = [item for item, keep in zip(Pop, next_gen) if keep]
    fitness = [item for item, keep in zip(fitness, next_gen) if keep]

    for i in range(len(population)):
        population[i]['fitness'] = fitness[i]

    return population

def truncation(objs, num):
    # truncate part of population
    npop = objs.shape[0]
    dis = squareform(pdist(objs))
    np.fill_diagonal(dis, np.inf)
    delete = np.full(npop, False)
    while np.sum(delete) < num:
        remain = np.where(~delete)[0]
        temp = np.sort(dis[remain][:, remain])
        delete[remain[np.argmin(temp[:, 0])]] = True
    return delete

def k_tournament_selection(pop, k, population_size):

    parent_pop = []

    for _ in range(population_size):
        # 随机选择k个参赛者（允许重复）
        tournament = random.sample(pop, k=k)
        # 找到适应度最小的个体（因为适应度越小越好）
        winner = min(tournament, key=lambda x: x.get('fitness', float('inf')))
        # 深拷贝获胜者并加入父代种群
        parent_pop.append(copy.deepcopy(winner))

    return parent_pop

def tensor_to_sdf(molecule_data, sdf_filename, atom_decoder,remove_hs=False,sanitize=False):

    # 从分子数据中提取相关信息
    x = molecule_data['x'].cpu().numpy()
    atom_type = molecule_data['atom_type'].cpu().numpy()
    node_mask = molecule_data['node_mask'].cpu().numpy().flatten()

    # 筛选有效原子
    valid_atoms = node_mask == 1
    x_valid = x[valid_atoms]
    atom_type_valid = atom_type[valid_atoms]

    # 创建分子对象
    mol = Chem.RWMol()

    # 添加原子
    atom_indices = []
    for atom_idx, atom_type_idx in enumerate(atom_type_valid):
        atom_symbol = atom_decoder[atom_type_idx]
        atom = Chem.Atom(atom_symbol)
        mol_idx = mol.AddAtom(atom)
        atom_indices.append(mol_idx)  # 记录原子在分子中的索引

    # 添加化学键
    n_atoms = len(atom_indices)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):  # 避免重复计算
            # 计算原子间距离
            distance = np.linalg.norm(x_valid[i] - x_valid[j])

            # 获取原子类型符号
            atom1_symbol = atom_decoder[atom_type_valid[i]]
            atom2_symbol = atom_decoder[atom_type_valid[j]]

            # 获取键级（调用用户提供的函数）
            bond_order = get_bond_order(atom1_symbol,
                                        atom2_symbol,
                                        distance,
                                        check_exists=True)

            # 添加化学键
            if bond_order > 0:
                bond_type = {  # 映射到RDKit键类型
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE
                }.get(bond_order, Chem.BondType.UNSPECIFIED)

                mol.AddBond(atom_indices[i],
                            atom_indices[j],
                            bond_type)

    # 设置构象坐标
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, x_valid[i].tolist())
    mol.AddConformer(conf)

    # 分子合法性检查（修复价态问题）
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # print(f"Sanitization warning: {str(e)}")
            return 0

    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol)  # RDKit的去氢方法
            nremoveH = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
            #sdf_filename = sdf_filename + 'removeH' + str(nremoveH)
            # print(f"Removed {sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)} hydrogen atoms")
        except Exception as e:
            # print(f"Hydrogen removal failed: {str(e)}")
            return 0
    sdf_filename = sdf_filename + '.sdf'
    # 保存SDF文件
    writer = Chem.SDWriter(sdf_filename)
    writer.write(mol)
    writer.close()
    return 1

def valOff(Parent,min_long,max_long,device,max_n_nodes,nodes_dist):
    Parentsize = int(round(len(Parent))/2)
    Off = []
    for i in range(Parentsize):
        if Parent[i+Parentsize]['T'] != Parent[i]['T']:
            assert print("noise not match!")
        offspring1, _ = crossover_continuous_unified(nodes_dist,Parent[i]['z'], Parent[i + Parentsize]['z'],
                                                          Parent[i]['long'], Parent[i + Parentsize]['long'],
                                                          min_long, max_long, device, max_n_nodes,Parent[i]['T'])
        offspring2, _ = crossover_continuous_unified(nodes_dist, Parent[i]['z'], Parent[i + Parentsize]['z'],
                                             Parent[i]['long'], Parent[i + Parentsize]['long'],
                                             min_long, max_long, device, max_n_nodes, Parent[i]['T'])
        Off.append(deepcopy(offspring1))
        Off.append(deepcopy(offspring2))
    return Off


def valOff_Patt(patt,Parent,min_long,max_long,device,max_n_nodes,nodes_dist):
    Parentsize = int(len(Parent))
    Off = []
    for i in range(Parentsize):
        if Parent[i]['T'] != patt['T']:
            assert print("noise not match!")
        offspring1, offspring2 = crossover_continuous_unified(nodes_dist, patt['z'], Parent[i]['z'],
                                                          patt['long'], Parent[i]['long'],
                                                          min_long, max_long, device, max_n_nodes,Parent[i]['T'],is_patt=True)
        Off.append(deepcopy(offspring1))
    return Off

def crossover_continuous_unified(nodes_dist, parent1, parent2, m, n, min_long, max_long, device, max_n_nodes,
                                 add_noise_step, is_patt=False):
    """
    统一的连续片段交叉函数
    :param is_patt: 如果为 True，则保留 parent1 的全部长度 (对应原来的 crossover_continuous_patt)
    """
    n_nodes_valid = nodes_dist.n_nodes

    # 1. 确定 k1 (来自父代1的长度)
    if is_patt:
        k1 = m
    else:
        k1_min = max(1, min_long - n)
        k1_max = min(m, max_long - 1)
        if k1_min > k1_max:
            return "无解"
        k1 = random.randint(k1_min, k1_max)

    # 2. 确定 k2 (来自父代2的长度)
    k2_min = max(1, min_long - k1)
    k2_max = min(n, max_long - k1)

    if k2_min > k2_max:
        return "无解"

    # 循环尝试直到总长度在合法节点分布内
    long = -1
    max_attempts = 50  # 防止死循环
    attempts = 0
    while long not in n_nodes_valid and attempts < max_attempts:
        k2 = random.randint(k2_min, k2_max)
        long = k1 + k2
        attempts += 1

    if attempts >= max_attempts:
        return "无解"

    # 3. 截取片段
    # 父代1
    start1 = torch.randint(0, m - k1 + 1, (1,)).item() if k1 < m else 0
    p1_selected = parent1[start1: start1 + k1]

    # 父代2
    start2 = torch.randint(0, n - k2 + 1, (1,)).item()
    p2_selected = parent2[start2: start2 + k2]

    # 4. 生成子代张量 (拼接并填充)
    # combined1: P1 + P2, combined2: P2 + P1
    combined_list = [
        torch.cat([p1_selected, p2_selected]),
        torch.cat([p2_selected, p1_selected])
    ]

    offspring_results = []
    node_mask = torch.zeros(max_n_nodes, 1)
    node_mask[0:long] = 1

    for combined in combined_list:
        gene = torch.zeros_like(parent1)
        gene[:long] = combined

        off_dict = {
            'z': gene.to(device),
            'long': long,
            'node_mask': node_mask.clone(),  # clone 避免引用冲突
            'T': add_noise_step,
            'EvaluatedPB': False,
            'EvaluatedVina': False,
            'EvaluatedSC': False
        }
        offspring_results.append(off_dict)

    return offspring_results[0], offspring_results[1]


@torch.no_grad()
def denoise_same_level(denoisepop, model, max_n_nodes, device, dataset_info, preds, evalcon,
                       context=None, iter_num=None):
    """
    升级版：完美兼容 batch 内个体拥有不同的噪声等级 (T)。
    动态切片推理，避免无意义的计算。
    """
    if not denoisepop:
        return denoisepop

    # 1. 提取每个个体的起始噪声步数 T
    T_starts = torch.tensor([item['T'] for item in denoisepop], dtype=torch.long, device=device)
    max_T = T_starts.max().item()

    if max_T == 0:
        # 如果大家都是 0，直接跳过去噪，直接解码即可
        pass

        # 2. 数据准备
    denoise_size = len(denoisepop)
    zs = torch.stack([item['z'] for item in denoisepop], dim=0).to(device)
    node_masks = torch.stack([item['node_mask'].squeeze(-1) for item in denoisepop], dim=0).to(device)
    node_masks_expanded = node_masks.unsqueeze(2)

    # 3. 掩码生成 (关键修改：保留 Batch 维度，不急着 view(-1, 1))
    edge_mask_batch = node_masks.unsqueeze(1) * node_masks.unsqueeze(2)
    diag_mask = ~torch.eye(max_n_nodes, device=device).bool().unsqueeze(0)
    edge_mask_batch = edge_mask_batch * diag_mask  # Shape:[Batch, N_nodes, N_nodes]

    # 4. 预处理
    zs = diffusion_utils.remove_mean_with_mask(zs, node_masks_expanded)
    if context is not None:
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_masks_expanded

    # =========================================================
    # 5. 核心采样循环 (支持不同 T 混合的动态子集推理)
    # =========================================================
    desc_str = f"EA Gen {iter_num} | Denoising" if iter_num is not None else "Denoising (Mixed T)"
    pbar = tqdm(reversed(range(max_T)), desc=desc_str, total=max_T, leave=False)

    for s in pbar:
        # 【核心逻辑】：找出在当前步数 s 处于活跃状态的个体
        # 比如某个个体 T=200，那么只有当 s < 200 (即 199 到 0) 时，它才参与去噪
        active_mask = T_starts > s

        if not active_mask.any():
            continue  # 如果当前没有任何个体活跃，直接跳过这一步

        active_idx = torch.where(active_mask)[0]

        # 从大 Batch 中切片出活跃的子集
        z_active = zs[active_idx]
        node_masks_active = node_masks_expanded[active_idx]
        # 对 edge_mask 进行切片后，再展平以满足模型的输入要求 [Sub_Batch * N * N, 1]
        edge_mask_active = edge_mask_batch[active_idx].view(-1, 1)
        context_active = context[active_idx] if context is not None else None

        # 计算活跃子集对应的时间步
        s_val = torch.full((len(active_idx), 1), s, device=device) / model.T
        t_val = s_val + (1.0 / model.T)

        # 仅将活跃子集送入模型进行去噪推理 (这比全量推理快得多！)
        z_updated = model.sample_p_zs_given_zt(
            s_val, t_val, z_active, node_masks_active, edge_mask_active, context_active, fix_noise=False
        )

        # 将更新后的 z 写回到总张量 zs 的对应位置中
        zs[active_idx] = z_updated

    # =========================================================
    # 6. 解码与种群更新
    # =========================================================
    # 解码前，将原本的 edge_mask_batch 整体展平
    edge_mask_flat = edge_mask_batch.view(-1, 1)

    if hasattr(model, 'vae'):
        x, h = model.vae.decode(zs, node_masks_expanded, edge_mask_flat, context)
    else:
        x, h = model.sample_p_xh_given_z0(zs, node_masks_expanded, edge_mask_flat, context)

    one_hot = h['categorical']
    for i, item in enumerate(denoisepop):
        item.update({
            'node_mask': node_masks[i].unsqueeze(-1),
            'one_hot': one_hot[i],
            'x': x[i],
            'T': 0, 'AddT': 0,
            'z': zs[i].detach(),
            'atom_type': one_hot[i].argmax(dim=-1)
        })

    # 7. 约束评价
    if evalcon:
        denoisepop = EvalPop_Con(denoisepop, dataset_info)
        denoisepop = EvalPop_Obj(denoisepop, device, preds, max_n_nodes)

    return denoisepop
# @torch.no_grad()
# def denoise_same_level(denoisepop, model, max_n_nodes, device, dataset_info, preds, evalcon,
#                        context=None, iter_num=None):
#     """
#     iter_num: 传入当前进化的代数 (int)
#     """
#     if not denoisepop:
#         return denoisepop
#
#     # 1. 数据准备
#     denoise_size = len(denoisepop)
#     start_step = denoisepop[0]['T']
#     zs = torch.stack([item['z'] for item in denoisepop], dim=0).to(device)
#     node_masks = torch.stack([item['node_mask'].squeeze(-1) for item in denoisepop], dim=0).to(device)
#
#     # 2. 掩码生成
#     edge_mask = node_masks.unsqueeze(1) * node_masks.unsqueeze(2)
#     diag_mask = ~torch.eye(max_n_nodes, device=device).bool().unsqueeze(0)
#     edge_mask = (edge_mask * diag_mask).view(-1, 1)
#     node_masks_expanded = node_masks.unsqueeze(2)
#
#     # 3. 预处理
#     zs = diffusion_utils.remove_mean_with_mask(zs, node_masks_expanded)
#     if context is not None:
#         context = context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_masks_expanded
#
#     # 4. 核心采样循环 (带代数显示的进度条)
#     # 动态构建描述文字
#     desc_str = f"EA Gen {iter_num} | Denoising" if iter_num is not None else "Denoising"
#
#     pbar = tqdm(reversed(range(start_step)), desc=desc_str, total=start_step, leave=False)
#
#     for s in pbar:
#         s_val = torch.full((denoise_size, 1), s, device=device) / model.T
#         t_val = s_val + (1.0 / model.T)
#         zs = model.sample_p_zs_given_zt(s_val, t_val, zs, node_masks_expanded, edge_mask, context, fix_noise=False)
#
#     # 5. 解码与种群更新 (保持原有逻辑)
#     if hasattr(model, 'vae'):
#         x, h = model.vae.decode(zs, node_masks_expanded, edge_mask, context)
#     else:
#         x, h = model.sample_p_xh_given_z0(zs, node_masks_expanded, edge_mask, context)
#
#     one_hot = h['categorical']
#     for i, item in enumerate(denoisepop):
#         item.update({
#             'node_mask': node_masks[i].unsqueeze(-1),
#             'one_hot': one_hot[i],
#             'x': x[i],
#             'T': 0, 'AddT': 0,
#             'atom_type': one_hot[i].argmax(dim=-1)
#         })
#
#     if evalcon:
#         denoisepop = EvalPop_Con(denoisepop, dataset_info)
#         denoisepop = EvalPop_Obj(denoisepop, device, preds, max_n_nodes)
#
#     return denoisepop

def draw_noised_pop(noisedpop,model,max_n_nodes,device,dataset_info,path):
    noised_size = len(noisedpop)
    node_masks = []
    zs = []
    for item in noisedpop:
        node_mask = item["node_mask"]
        z = item['z']
        zs.append(z)
        # 去掉最后一个维度（从形状 [29, 1] 变为 [29]）
        node_mask = node_mask.squeeze(-1)
        node_masks.append(node_mask)

    # 将所有张量组合成一个新的张量，形状为 [k, 29]
    node_masks = torch.stack(node_masks, dim=0)
    zs = torch.stack(zs, dim=0)
    edge_mask = node_masks.unsqueeze(1) * node_masks.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask.to(edge_mask.device)
    edge_mask = edge_mask.view(noised_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_masks = node_masks.unsqueeze(2).to(device)
    zs = zs.to(device)
    zs = diffusion_utils.remove_mean_with_mask(zs, node_masks)

    if hasattr(model, 'vae'):
        x, h = model.vae.decode(zs, node_masks, edge_mask, None)
    else:
        x, h = model.sample_p_xh_given_z0(zs, node_masks, edge_mask, None)
    one_hot = h['categorical']

    denoisedpop = deepcopy(noisedpop)

    for i in range(noised_size):
        denoisedpop[i]['one_hot'] = one_hot[i]
        denoisedpop[i]['x'] = x[i]
        denoisedpop[i]['atom_type'] = torch.argmax(one_hot[i], dim=1)
        save3D(denoisedpop[i],dataset_info,path,i)




@torch.no_grad()
def add_noise(generative_model, Pop, device):
    NPop = len(Pop)
    for i in range(NPop):
        z = Pop[i]['z']
        x = z[:,:3]
        node_mask = Pop[i]['node_mask']
        added_noise_step = Pop[i]['AddT']
        t_int = torch.randint(added_noise_step, added_noise_step + 1, size=(1, 1), device=x.device).float()
        t = t_int / generative_model.T
        gamma_t = generative_model.inflate_batch_array(generative_model.gamma(t), x)
        alpha_t = generative_model.alpha(gamma_t, x)
        sigma_t = generative_model.sigma(gamma_t, x)
        x = x.unsqueeze(0)
        node_mask = node_mask.unsqueeze(0)

        eps = generative_model.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask).to(device)

        z_t = alpha_t * z + sigma_t * eps
        # if added_noise_step >= 990:
        #     # 彻底抹杀父代信号，模拟纯 InitPop 的物理状态
        #     z_t = sigma_t * eps
        # else:
        #     z_t = alpha_t * z + sigma_t * eps
        z_t = z_t.squeeze(0)
        Pop[i]['z'] = z_t.to(device)
        Pop[i]['noisedxh'] = z_t.to(device)
        Pop[i]['T'] = added_noise_step
        Pop[i]['AddT'] = 0
        Pop[i]['node_mask'] = Pop[i]['node_mask'].detach().cpu()

    return Pop



def FNDS(PopObj, PopCon=None):
    """
    Calculate the fitness of each solution
    """
    N = PopObj.shape[0]

    if PopCon is None:
        CV = np.zeros(N)
    else:
        CV = np.maximum(0, PopCon)

    # Detect the dominance relation between each two solutions
    Dominate = np.zeros((N, N), dtype=bool)
    for i in range(N - 1):
        for j in range(i + 1, N):
            if CV[i] < CV[j]:
                Dominate[i, j] = True
            elif CV[i] > CV[j]:
                Dominate[j, i] = True
            else:
                k = int(np.any(PopObj[i] < PopObj[j])) - int(np.any(PopObj[i] > PopObj[j]))
                if k == 1:
                    Dominate[i, j] = True
                elif k == -1:
                    Dominate[j, i] = True

    # Calculate S(i)
    S = np.sum(Dominate, axis=1)

    # Calculate R(i)
    R = np.zeros(N)
    for i in range(N):
        R[i] = np.sum(S[Dominate[:, i]])

    # Calculate D(i)
    Distance = np.sqrt(np.sum((PopObj[:, np.newaxis, :] - PopObj[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(Distance, np.inf)
    Distance.sort(axis=1)
    D = 1 / (Distance[:, int(np.sqrt(N)) - 1] + 2)

    # Calculate the fitnesses
    Fitness = R + D
    return Fitness



def _to_cpu(mol):
    """
    辅助函数：将分子字典中的所有 Tensor 转移到 CPU。
    避免多进程 pickle GPU Tensor 导致的崩溃。
    """
    new_mol = {}
    for k, v in mol.items():
        if isinstance(v, torch.Tensor):
            new_mol[k] = v.detach().cpu()
        else:
            new_mol[k] = v
    return new_mol


def _extract_matching_graph(mol, dataset_info):
    """Build a bond graph and retain local-to-padded atom index mapping."""
    mask = mol['node_mask'].squeeze(-1) == 1
    valid_indices = torch.nonzero(mask).flatten().detach().cpu().numpy()
    coordinates = mol['x'][mask].detach().cpu().numpy()
    type_indices = mol['atom_type'][mask].detach().cpu().numpy()
    atom_types = [
        dataset_info['atom_decoder'][int(type_index)]
        for type_index in type_indices
    ]
    return (
        get_bond_mat(coordinates, atom_types),
        atom_types,
        valid_indices,
    )


def _match_pattern_collection(mol, patt_list, dataset_info, exclusive=True):
    """Jointly match patterns and return per-pattern missing-atom percentages.

    When ``exclusive`` is true, one target atom may belong to at most one
    pattern.  Complete embeddings are attempted first; if no joint complete
    assignment exists, connected partial embeddings are optimized globally.
    """
    target_adj, target_types, target_valid_indices = _extract_matching_graph(
        mol, dataset_info
    )
    fragment_graphs = [
        _extract_matching_graph(pattern, dataset_info)[:2]
        for pattern in patt_list
    ]
    fragment_sizes = [len(fragment_types) for _, fragment_types in fragment_graphs]

    full_option_lists = [
        _full_match_options(
            fragment_adj,
            fragment_types,
            target_adj,
            target_types,
        )
        if fragment_types
        else []
        for fragment_adj, fragment_types in fragment_graphs
    ]

    if exclusive:
        selected = _choose_disjoint_options(
            full_option_lists,
            [max(1, size) for size in fragment_sizes],
        )
    elif all(full_option_lists):
        selected = [options[0] for options in full_option_lists]
    else:
        selected = None

    if selected is None:
        option_lists = []
        for (fragment_adj, fragment_types), full_options in zip(
            fragment_graphs, full_option_lists
        ):
            options = _connected_match_options(
                fragment_adj,
                fragment_types,
                target_adj,
                target_types,
                full_options=full_options,
            )
            # An empty match keeps the global optimizer defined even when a
            # pattern has no atom type in common with the target.
            options.append((0, frozenset(), {}))
            option_lists.append(options)

        if exclusive:
            selected = _choose_disjoint_options(
                option_lists,
                [max(1, size) for size in fragment_sizes],
            )
        else:
            selected = [options[0] for options in option_lists]

    if selected is None:
        selected = [(0, frozenset(), {}) for _ in patt_list]

    raw_scores = []
    matched_local_indices = set()
    for fragment_size, option in zip(fragment_sizes, selected):
        matched_size = option[0]
        raw_scores.append(
            100.0
            if fragment_size == 0
            else 100.0 * (1.0 - matched_size / fragment_size)
        )
        matched_local_indices.update(option[1])

    matched_global_indices = [
        int(target_valid_indices[index])
        for index in sorted(matched_local_indices)
    ]
    return raw_scores, matched_global_indices


def _evaluate_single_molecule_patt(mol, patt_list, dataset_info, tolerance, exclusive=True):
    """
    Worker 函数：处理单个分子的多片段匹配逻辑。
    必须定义在顶层以支持多进程序列化。
    """
    num_patterns = len(patt_list)
    # --- A. 计算原始匹配得分 ---
    if mol.get('Constraint', 0) == 0:
        raw_scores, all_matched_indices = _match_pattern_collection(
            mol,
            patt_list,
            dataset_info,
            exclusive=exclusive,
        )
    else:
        raw_scores = [100.0] * num_patterns
        all_matched_indices = []

    # --- B. 应用 ϵ-约束 (Tolerance) ---
    final_con_list = []
    fake_sc_list = []

    for sc in raw_scores:
        if tolerance > sc:
            final_con_list.append(0.0)  # 伪可行
            fake_sc_list.append(sc if sc > 0 else 0.0)
        else:
            final_con_list.append(sc)  # 实际惩罚
            fake_sc_list.append(0.0)

    # 去重并排序匹配索引
    unique_matched_indices = sorted(list(set(all_matched_indices)))

    # 返回结果字典
    return {
        'structure_Con': final_con_list,
        'matched_indices': unique_matched_indices,
        'FakeSC': fake_sc_list if any(x > 0 for x in fake_sc_list) else None
    }


def EvalPop_Patt_MP(
    Pop,
    patt,
    dataset_info,
    tolerance,
    num_workers=8,
    exclusive=True,
):
    """
    多进程并行版：评价种群相对于参考片段的结构约束。跳过已评估（EvaluatedSC=True）的分子。
    """
    # 1. 基础检查：无约束时直接赋值默认值
    if patt is None:
        for mol in Pop:
            mol['structure_Con'] = [0.0]
            mol['matched_indices'] = []
            mol.pop('FakeSC', None)
            mol['EvaluatedSC'] = True
        return Pop

    # 2. 统一约束格式（支持单片段或多片段列表）
    if isinstance(patt, list):
        patt_list = patt
    else:
        patt_list = [patt]

    # 将参考片段转为 CPU 格式（VF2 匹配需要在 CPU 上进行）
    cpu_patt_list = [_to_cpu(p) for p in patt_list]

    # 3. 准备 worker 函数（使用 partial 固定公共参数）
    worker_func = partial(
        _evaluate_single_molecule_patt,
        patt_list=cpu_patt_list,
        dataset_info=dataset_info,
        tolerance=tolerance,
        exclusive=exclusive,
    )

    # 4. 筛选出未评估的分子及其索引
    indices_to_process = []
    molecules_to_process = []  # 存储 CPU 版本的分子数据
    for i, mol in enumerate(Pop):
        if not mol.get('EvaluatedSC', False):   # 默认 False，表示需要评估
            # 将分子数据转移到 CPU（假设 _to_cpu 已定义）
            cpu_mol = _to_cpu(mol)
            indices_to_process.append(i)
            molecules_to_process.append(cpu_mol)

    # 如果没有需要评估的分子，直接返回
    if not indices_to_process:
        print("All molecules already evaluated, skipping pattern matching.")
        return Pop

    # 5. 执行计算（并行或串行）
    results = []
    if num_workers > 1 and len(molecules_to_process) > 1:
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(worker_func, molecules_to_process),
                    total=len(molecules_to_process),
                    desc=f"Multi-Pattern Matching (Jobs={num_workers})",
                    leave=False
                ))
        except Exception as e:
            print(f"Warning: Multiprocessing failed ({e}), falling back to serial execution.")
            # 回退到串行处理
            for mol in tqdm(molecules_to_process, desc="Fallback Serial Matching", leave=False):
                results.append(worker_func(mol))
    else:
        # 串行模式（单进程或调试）
        for mol in tqdm(molecules_to_process, desc="Serial Pattern Matching", leave=False):
            results.append(worker_func(mol))

    # 6. 将结果回填到原始种群，并标记为已评估
    for idx, res in zip(indices_to_process, results):
        Pop[idx]['structure_Con'] = res['structure_Con']
        Pop[idx]['matched_indices'] = res['matched_indices']
        if res.get('FakeSC') is not None:
            Pop[idx]['FakeSC'] = res['FakeSC']
        else:
            Pop[idx].pop('FakeSC', None)
        Pop[idx]['EvaluatedSC'] = True      # 标记评估完成

    return Pop


def EvalPop_Patt(Pop, patt, dataset_info, tolerance, exclusive=True):
    """
    评价种群相对于一个或多个参考片段的结构约束，并保存匹配索引。
    """
    # 1. 如果没有参考片段
    if patt is None:
        for mol in Pop:
            mol['structure_Con'] = [0.0]
            mol['matched_indices'] = []  # 初始化为空
            mol.pop('FakeSC', None)
            mol['EvaluatedSC'] = True
        return Pop

    # 2. 统一输入格式
    if isinstance(patt, list):
        patt_list = patt
    else:
        patt_list = [patt]

    # 3. 使用 tqdm 包装循环
    pbar = tqdm(Pop, desc="Multi-Pattern Matching", leave=False)

    for mol in pbar:
        if mol.get('EvaluatedSC', False):
            continue
        result = _evaluate_single_molecule_patt(
            mol,
            patt_list,
            dataset_info,
            tolerance,
            exclusive=exclusive,
        )
        mol['structure_Con'] = result['structure_Con']
        mol['matched_indices'] = result['matched_indices']
        mol['EvaluatedSC'] = True
        if result.get('FakeSC') is not None:
            mol['FakeSC'] = result['FakeSC']
        else:
            mol.pop('FakeSC', None)

    return Pop


@torch.no_grad()
def Get_Fitness_Pareto(Pop, dataset_info, device, preds, n_nodes, ObjName, patt, tolerance, minimize=True,
                       num_workers=1, calCon=True):

    if calCon:
        Pop = EvalPop_Con(Pop, dataset_info)
        Pop = EvalPop_Patt_MP(Pop, patt, dataset_info, tolerance, num_workers=num_workers)
        Pop = EvalPop_Obj(Pop, device, preds, n_nodes)

    Obj = []
    Con = []

    # 2. 提取目标值 (Objectives)
    for j in range(len(Pop)):
        res = []
        for obj in ObjName:
            val = Pop[j][obj].item() if isinstance(Pop[j][obj], torch.Tensor) else Pop[j][obj]
            if minimize:
                res.append(val)
            else:
                res.append(-val)
        Obj.append(deepcopy(res))
    Obj = np.array(Obj)

    # 3. 提取并汇总约束值 (Constraints)
    for i in range(len(Pop)):
        # A. 化学稳定性约束 (硬约束，必须遵守)
        # max(0, con) 确保只有违背时才为正数
        chem_con = max(0, Pop[i].get('Constraint', 0))

        # B. 结构片段约束 (结构任务约束)
        struct_con_data = Pop[i].get('structure_Con', [0])
        if isinstance(struct_con_data, list):
            struct_con_sum = sum(struct_con_data)
        else:
            struct_con_sum = struct_con_data

        struct_con_sum = max(0, struct_con_sum)

        # C. 总约束违背 (Total CV)
        # 这里必须包含化学约束，这是前提
        total_cv = chem_con + struct_con_sum
        Con.append(total_cv)

    Con = np.array(Con)

    # 4. 基于约束的快速非支配排序 (FNDS)
    Fitness = FNDS(Obj, Con)

    # 5. 赋值 Fitness
    for i in range(len(Pop)):
        Pop[i]['fitness'] = Fitness[i]
    return Pop



@torch.no_grad()
def EvalPop_Con(Pop,dataset_info):
    #molecules = {key: Pop[key].unsqueeze(0) for key in Pop}
    for i in range(len(Pop)):
        molecules = {key: Pop[i][key].unsqueeze(0) for key in Pop[i] if isinstance(Pop[i][key], torch.Tensor)}
        validity_dict, rdkit_tuple = analyze.analyze_stability_for_molecules(molecules, dataset_info)
        Pop[i]['atm_stable'] = validity_dict['atm_stable']
        Pop[i]['mol_stable'] = validity_dict['mol_stable']
        Pop[i]['validity_rdkit'] =  rdkit_tuple[0][0]
        Pop[i]['uniqueness_rdkit'] = rdkit_tuple[0][1]
        Pop[i]['novelty_rdkit'] = rdkit_tuple[0][2]
        Pop[i]['valid_SMILES_rdkit'] = rdkit_tuple[1][0] if rdkit_tuple[1] is not None else 'None'
        Pop[i]['Constraint'] = (1-Pop[i]['atm_stable']) + (1-Pop[i]['mol_stable']) + (1-Pop[i]['validity_rdkit']) * 1000
    return Pop


@torch.no_grad()
def Get_Pred(Obj,device):
    pred_add = './qm9/property_prediction/outputs/'
    pred = {}
    for i in range(len(Obj)):
        model = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=128, device=device, n_layers=7, coords_weight=1.0, attention=1, node_attr=0)
        model_add = os.path.join(pred_add,Obj[i]) + '/best_checkpoint.npy'
        flow_state_dict = torch.load(model_add,map_location=device)
        model.load_state_dict(flow_state_dict)
        model.eval()
        pred[Obj[i]] = deepcopy(model)
    return pred

@torch.no_grad()
def EvalPop_Obj(molecule, device, preds, n_nodes):
    if not preds:
        return molecule
    NPop = len(molecule)
    data = {}
    for item in molecule:
        for key, value in item.items():
            if isinstance(value, torch.Tensor):  # 检查是否为tensor类型
                if key not in data:  # 如果键不存在于data字典中，初始化为一个空列表
                    data[key] = []
                data[key].append(value)  # 将tensor添加到对应键的列表中

    # 将每个键对应的tensor列表拼接
    for key in data.keys():
        data[key] = torch.stack(data[key], dim=0)

    atom_positions = data['x'].view(NPop * n_nodes, -1).to(device, torch.float32)
    atom_mask = data['node_mask'].view(NPop * n_nodes, -1).to(device, torch.float32)

    temp_nm = data['node_mask'].squeeze(2)
    edge_mask = temp_nm.unsqueeze(1) * temp_nm.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(edge_mask.device)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(NPop * n_nodes * n_nodes, 1).to(device, torch.float32)

    nodes = data['one_hot'].to(device, torch.float32)
    nodes = nodes.view(NPop * n_nodes, -1)
    # nodes = torch.cat([one_hot, charges], dim=1)
    edges = prop_utils.get_adj_matrix(n_nodes, NPop, device)
    for key in preds:
        model = preds[key]
        res = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        mean = property_mean[key]
        mad = property_mad[key]
        conv = conversion[key]
        res = mad * res + mean
        res = res * conv
        for j in range(NPop):
            molecule[j][key] = res[j]
    return molecule


def EnvironmentalSelectionSingle(Pop, N):
    # 提取适应度值
    fitness = np.array([individual['fitness'] for individual in Pop])

    # 初始化下一代标记数组
    next_gen = np.zeros(len(Pop), dtype=bool)

    # 环境选择逻辑
    # 假设是极小化问题，适应度值越小越好
    sorted_indices = np.argsort(fitness)  # 升序排序索引
    next_gen[sorted_indices[:N]] = True

    # 构造下一代种群
    population = [Pop[i] for i in range(len(Pop)) if next_gen[i]]

    return population


@contextmanager
def suppress_stdout_stderr():
    """彻底屏蔽 stdout 和 stderr"""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
def optimize_population_geometry(Pop, device, fmax=0.05, max_steps=100):
    # 统计有效且需要优化的分子数量
    valid_candidates = sum(1 for m in Pop if m.get('validity_rdkit', 0) == 1)
    print(f"\n>>> Starting Geometry Optimization for {valid_candidates} valid molecules...")

    success_cnt = 0
    skipped_cnt = 0

    # 这里的 tqdm 只显示有效分子的进度会更直观，但为了索引方便，我们还是遍历全部
    for i, mol in enumerate(tqdm(Pop, desc="Optimizing")):

        # --- 修改 1: 筛选条件 ---
        # 只有当 validity_rdkit 存在且为 1 (或True) 时才做优化
        # 如果你存储的是布尔值，直接用 if not mol.get('validity_rdkit'):
        if mol.get('validity_rdkit', 0) != 1:
            mol['opt_converged'] = False
            mol['opt_energy'] = 1e9
            skipped_cnt += 1
            continue

        try:
            n_atoms = int(mol['long'])
            pos = mol['x'][:n_atoms].detach().cpu().numpy()
            numbers = mol['atomic_numbers'][:n_atoms].detach().cpu().numpy()

            atoms = Atoms(numbers=numbers, positions=pos)

            # --- 修改 2: 让 TBLite 闭嘴 ---
            # 加上 verbosity=0
            atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)

            # 使用上下文管理器屏蔽 BFGS 和底层库的所有输出
            with suppress_stdout_stderr():
                # logfile=None 屏蔽 ASE 的 python 层输出
                opt = BFGS(atoms, logfile=None)
                converged = opt.run(fmax=fmax, steps=max_steps)

            final_energy = atoms.get_potential_energy()

            # 简单的物理过滤
            if not converged or final_energy > 0:
                mol['opt_converged'] = False
                mol['opt_energy'] = 1e9
            else:
                mol['x'][:n_atoms] = torch.tensor(atoms.get_positions(), device=device, dtype=torch.float)
                mol['opt_energy'] = final_energy
                mol['opt_converged'] = True
                success_cnt += 1

        except Exception:
            mol['opt_converged'] = False
            mol['opt_energy'] = 1e9

    print(
        f">>> Finished. Optimized: {success_cnt}, Skipped(Invalid): {skipped_cnt}, Failed: {len(Pop) - success_cnt - skipped_cnt}")
    return Pop



def benchmark_xtb_final(loader, device, num_samples=100):
    # 常数
    HA_TO_EV = 27.211386
    BOHR_TO_ANG = 0.529177
    AU_TO_DEBYE = 2.541746
    ATOM_MAP = np.array([1, 6, 7, 8, 9])

    props = ['homo', 'lumo', 'gap', 'mu']
    results = {p: {'pred': [], 'gt': []} for p in props}

    print(f"\n>>> Starting Final Benchmark (Dynamic Occupation Check)...")

    count = 0
    for batch_idx, data in enumerate(loader):
        if count >= num_samples: break
        batch_size = data['num_nodes'].size(0) if 'num_nodes' in data else data['positions'].shape[0]

        for i in range(batch_size):
            if count >= num_samples: break
            try:
                # --- 1. 数据准备 (保持不变，因为 Mu 也是对的) ---
                if 'atom_mask' in data:
                    n_valid = int(data['atom_mask'][i].sum().item())
                else:
                    n_valid = data['positions'].shape[1]

                pos_ang = data['positions'][i][:n_valid].detach().cpu().numpy()
                pos_bohr = pos_ang / BOHR_TO_ANG

                if 'one_hot' in data:
                    one_hot = data['one_hot'][i][:n_valid].detach().cpu().numpy()
                    z = ATOM_MAP[np.argmax(one_hot, axis=1)]
                else:
                    z = data['charges'][i][:n_valid].squeeze().detach().cpu().numpy().astype(int)

                # --- 2. xTB 计算 ---
                z = np.ascontiguousarray(z, dtype=np.int32)
                pos_bohr = np.ascontiguousarray(pos_bohr, dtype=np.float64)
                calc = Calculator(Param.GFN2xTB, z, pos_bohr)
                calc.set_verbosity(VERBOSITY_MUTED)
                res = calc.singlepoint()

                # --- 3. 核心修复：基于占据数找 HOMO/LUMO ---
                eigenvalues = np.array(res.get_orbital_eigenvalues())  # Hartree
                occupations = np.array(res.get_orbital_occupations())  # 0.0 ~ 2.0

                # 寻找 HOMO 索引：最后一个占据数 > 0.1 的轨道
                # (QM9 是闭壳层，占据数通常是 2.0，0.1 是安全阈值)
                occupied_indices = np.where(occupations > 0.1)[0]

                if len(occupied_indices) == 0:
                    # 异常保护
                    print("Warning: No occupied orbitals found?")
                    continue

                homo_idx = occupied_indices[-1]  # 占据轨道的最后一个
                lumo_idx = homo_idx + 1  # 下一个就是 LUMO

                # 提取能量 (Hartree)
                homo_ha = eigenvalues[homo_idx]

                # 保护：防止 LUMO 索引越界 (虽然不太可能)
                if lumo_idx < len(eigenvalues):
                    lumo_ha = eigenvalues[lumo_idx]
                else:
                    lumo_ha = homo_ha + 0.1  # 这种情况极少见

                gap_ha = lumo_ha - homo_ha

                # 转换单位 (Hartree -> eV)
                preds = {
                    'homo': homo_ha * HA_TO_EV,
                    'lumo': lumo_ha * HA_TO_EV,
                    'gap': gap_ha * HA_TO_EV,
                    'mu': np.linalg.norm(res.get_dipole()) * AU_TO_DEBYE
                }

                # --- 4. 获取 GT (保持不变) ---
                gt_values = {}
                for key in props:
                    val = np.nan
                    if key in data:
                        val = data[key][i].item()
                    elif key.upper() in data:
                        val = data[key.upper()][i].item()
                    gt_values[key] = val

                # 记录
                for key in props:
                    if not np.isnan(gt_values[key]):
                        results[key]['pred'].append(preds[key])
                        results[key]['gt'].append(gt_values[key])

                count += 1
                print(f"\rProgress: {count}/{num_samples}", end="")

            except Exception as e:
                # print(e)
                pass

    # --- 统计 ---
    print("\n\n" + "=" * 65)
    print(f"{'Property':<10} | {'MAE':<10} | {'R2':<10} | {'Pearson':<10}")
    print("-" * 65)

    for key in props:
        p = np.array(results[key]['pred'])
        g = np.array(results[key]['gt'])
        if len(p) > 1:
            mae = np.mean(np.abs(p - g))
            r2 = r2_score(g, p)
            pearson, _ = pearsonr(g, p)
            print(f"{key:<10} | {mae:<10.4f} | {r2:<10.4f} | {pearson:<10.4f}")
    print("=" * 65)


def EvalPop_Obj_XTB(molecule, dataset_info, target_props):
    """
    使用 xTB (GFN2-xTB) 计算种群的物理属性。
    替代/补充 EvalPop_Obj，不依赖神经网络，而是实时物理计算。

    Args:
        molecule: 分子列表 (List of dicts)
        dataset_info: 包含原子映射信息的字典
        target_props: 需要计算的属性列表，支持 ['mu_xtb', 'homo_xtb', 'lumo_xtb', 'gap_xtb']
    """
    if not target_props:
        return molecule

    # --- 1. 物理常数 (与Benchmark一致) ---
    HA_TO_EV = 27.211386
    BOHR_TO_ANG = 0.529177
    AU_TO_DEBYE = 2.541746

    # 准备原子映射: 假设 dataset_info['atom_decoder'] = ['H', 'C', 'N', 'O', 'F']
    # 建立 索引 -> 原子序数(Z) 的映射表
    symbol_to_z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17}
    decoder = dataset_info['atom_decoder']
    # 例如: [1, 6, 7, 8, 9]
    atom_map = np.array([symbol_to_z[sym] for sym in decoder], dtype=np.int32)

    # --- 2. 遍历种群逐个计算 ---
    # xTB 是 CPU 密集的，且 C++ 接口不支持 Batch，只能循环
    # 为了进度可视化，这里可以加 tqdm，但为了函数纯净性我先不加

    for i, mol in enumerate(molecule):
        try:
            # --- A. 数据提取与预处理 ---
            # 1. 获取有效原子数
            # node_mask 形状通常是 [N, 1] 或 [N]
            n_atoms = int(mol['node_mask'].sum().item())

            # 2. 提取坐标 (Å -> Bohr)
            # x 形状: [max_nodes, 3] -> 切片 -> numpy
            pos_ang = mol['x'][:n_atoms].detach().cpu().numpy().astype(np.float64)
            pos_bohr = pos_ang / BOHR_TO_ANG

            # 3. 提取原子序数 Z
            if 'atom_type' in mol:
                # 如果已经有 argmax 后的类型索引
                indices = mol['atom_type'][:n_atoms].detach().cpu().numpy()
                z = atom_map[indices]
            elif 'one_hot' in mol:
                # 如果是 one_hot [max_nodes, n_types]
                one_hot = mol['one_hot'][:n_atoms].detach().cpu().numpy()
                indices = np.argmax(one_hot, axis=1)
                z = atom_map[indices]
            else:
                raise ValueError("Molecule dict missing 'atom_type' or 'one_hot'")

            # 确保内存连续 (C++ 接口要求)
            z = np.ascontiguousarray(z, dtype=np.int32)
            pos_bohr = np.ascontiguousarray(pos_bohr, dtype=np.float64)

            # --- B. xTB 计算核心 ---
            calc = Calculator(Param.GFN2xTB, z, pos_bohr)
            calc.set_verbosity(VERBOSITY_MUTED)  # 静默模式
            res = calc.singlepoint()  # 执行 SCF 计算

            # --- C. 属性提取 (复用验证过的逻辑) ---
            # 1. 偶极矩 (Debye)
            mu_val = np.nan
            if any(k in target_props for k in ['mu_xtb']):
                dipole_vec = res.get_dipole()
                mu_val = np.linalg.norm(dipole_vec) * AU_TO_DEBYE

            # 2. 轨道能量 (eV) - 动态占据逻辑
            homo_val = np.nan
            lumo_val = np.nan
            gap_val = np.nan

            need_orbital = any(k in target_props for k in ['homo_xtb', 'lumo_xtb', 'gap_xtb'])

            if need_orbital:
                eigenvalues = np.array(res.get_orbital_eigenvalues())  # Hartree
                occupations = np.array(res.get_orbital_occupations())  # 0 ~ 2

                # 寻找 HOMO: 最后一个占据数 > 0.1 的轨道
                occ_indices = np.where(occupations > 0.1)[0]

                if len(occ_indices) > 0:
                    homo_idx = occ_indices[-1]
                    lumo_idx = homo_idx + 1

                    # 提取并转 eV
                    homo_val = eigenvalues[homo_idx] * HA_TO_EV

                    if lumo_idx < len(eigenvalues):
                        lumo_val = eigenvalues[lumo_idx] * HA_TO_EV
                    else:
                        lumo_val = homo_val + 0.1  # 极罕见情况兜底

                    gap_val = lumo_val - homo_val

            # --- D. 写回分子字典 ---
            # 结果转为 Tensor 以保持与原有流程兼容 (float32)
            if 'mu_xtb' in target_props:
                mol['mu_xtb'] = torch.tensor(mu_val, dtype=torch.float32)
            if 'homo_xtb' in target_props:
                mol['homo_xtb'] = torch.tensor(homo_val, dtype=torch.float32)
            if 'lumo_xtb' in target_props:
                mol['lumo_xtb'] = torch.tensor(lumo_val, dtype=torch.float32)
            if 'gap_xtb' in target_props:
                mol['gap_xtb'] = torch.tensor(gap_val, dtype=torch.float32)

        except Exception as e:
            # --- E. 异常处理 ---
            # 如果 SCF 不收敛或几何结构太差，填入 NaN 或极差的值
            # print(f"xTB Failed for mol {i}: {e}")
            nan_tensor = torch.tensor(float('nan'), dtype=torch.float32)
            for key in target_props:
                mol[key] = nan_tensor

    return molecule



def benchmark_xtb_hybrid_opt(loader, device, num_samples=100):
    # --- 1. 常数 ---
    HA_TO_EV = 27.211386
    ATOM_MAP = np.array([1, 6, 7, 8, 9])
    AU_TO_DEBYE = 2.541746
    BOHR_TO_ANG = 0.529177

    props = ['homo', 'lumo', 'gap', 'mu']
    results = {p: {'pred': [], 'gt': []} for p in props}

    print(f"\n>>> Starting Benchmark (Hybrid: ASE Optimization + XTB Calculation)...")

    count = 0
    for batch_idx, data in enumerate(loader):
        if count >= num_samples: break
        batch_size = data['num_nodes'].size(0) if 'num_nodes' in data else data['positions'].shape[0]

        for i in range(batch_size):
            if count >= num_samples: break
            try:
                # --- A. 数据准备 ---
                if 'atom_mask' in data:
                    n_valid = int(data['atom_mask'][i].sum().item())
                else:
                    n_valid = data['positions'].shape[1]

                # 原始坐标 (Angstrom)
                pos_ang = data['positions'][i][:n_valid].detach().cpu().numpy()

                # 原子序数
                if 'one_hot' in data:
                    one_hot = data['one_hot'][i][:n_valid].detach().cpu().numpy()
                    z = ATOM_MAP[np.argmax(one_hot, axis=1)]
                else:
                    z = data['charges'][i][:n_valid].squeeze().detach().cpu().numpy().astype(int)

                # --- B. 步骤1: 使用 ASE + tblite 进行几何优化 ---
                mol = Atoms(numbers=z, positions=pos_ang)
                mol.calc = ASETBLite(method="GFN2-xTB")

                # 运行优化 (BFGS)
                # 这一步会修改 mol 的坐标到能量极小值
                opt = BFGS(mol, logfile=None)
                opt.run(fmax=0.05)

                # 获取优化后的坐标 (Angstrom)
                opt_pos = mol.get_positions()

                # --- C. 步骤2: 使用 xtb-python 计算详细属性 ---
                # 这一步我们切回你最熟悉的 xtb 接口

                # 1. 转换坐标 Angstrom -> Bohr
                opt_pos_bohr = opt_pos / BOHR_TO_ANG

                # 2. 类型转换 (C++ 接口要求)
                z_calc = np.ascontiguousarray(z, dtype=np.int32)
                pos_calc = np.ascontiguousarray(opt_pos_bohr, dtype=np.float64)

                # 3. 计算
                calc = XtbCalculator(Param.GFN2xTB, z_calc, pos_calc)
                calc.set_verbosity(VERBOSITY_MUTED)
                res = calc.singlepoint()

                # --- D. 提取属性 (复用你验证成功的逻辑) ---
                # 单位: Hartree
                eigenvalues = np.array(res.get_orbital_eigenvalues())
                occupations = np.array(res.get_orbital_occupations())

                # 找 HOMO
                occ_indices = np.where(occupations > 0.1)[0]
                if len(occ_indices) == 0: continue

                homo_idx = occ_indices[-1]
                lumo_idx = homo_idx + 1

                homo_ha = eigenvalues[homo_idx]
                lumo_ha = eigenvalues[lumo_idx] if lumo_idx < len(eigenvalues) else homo_ha + 0.1
                gap_ha = lumo_ha - homo_ha

                # 偶极矩 (xtb 返回 e*Bohr) -> 转 Debye
                mu_debye = np.linalg.norm(res.get_dipole()) * AU_TO_DEBYE

                preds = {
                    'homo': homo_ha * HA_TO_EV,  # 转 eV
                    'lumo': lumo_ha * HA_TO_EV,  # 转 eV
                    'gap': gap_ha * HA_TO_EV,  # 转 eV
                    'mu': mu_debye
                }

                # --- E. 获取 GT ---
                gt_values = {}
                for key in props:
                    val = np.nan
                    if key in data:
                        val = data[key][i].item()
                    elif key.upper() in data:
                        val = data[key.upper()][i].item()
                    gt_values[key] = val

                # 记录
                for key in props:
                    if not np.isnan(gt_values[key]):
                        results[key]['pred'].append(preds[key])
                        results[key]['gt'].append(gt_values[key])

                count += 1
                print(f"\rProgress: {count}/{num_samples}", end="")

            except Exception as e:
                # 优化失败或SCF不收敛则跳过
                pass

    # --- F. 统计 ---
    print("\n\n" + "=" * 65)
    print(f"{'Property':<10} | {'MAE':<10} | {'R2':<10} | {'Pearson':<10}")
    print("-" * 65)

    for key in props:
        p = np.array(results[key]['pred'])
        g = np.array(results[key]['gt'])
        if len(p) > 1:
            mae = np.mean(np.abs(p - g))
            r2 = r2_score(g, p)
            pearson, _ = pearsonr(g, p)
            print(f"{key:<10} | {mae:<10.4f} | {r2:<10.4f} | {pearson:<10.4f}")
    print("=" * 65)



import numpy as np
import torch
from collections import deque
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

class AdaptiveNoiseScheduler:
    def __init__(self, dataset_info,
                 initial_noise=500, min_noise=50, max_noise=1000,
                 step_size=30,
                 drop_factor=0.5,
                 use_hv_trigger=False, cent_patience=3,
                 penalty_factor=1, mode='qm9', jittor_scale=10, beta=2):

        self.dataset_info = dataset_info
        self.min_t = min_noise
        self.max_t = max_noise
        self.step_size = step_size
        self.drop_factor = drop_factor
        self.penalty_factor = penalty_factor # 调节倾向：越大越倾向于找拐点，越小越倾向于找最大值
        self.mode = mode
        self.jittor_scale = jittor_scale
        self.beta = beta
        
        # 当前噪声
        self.curr_noise = initial_noise

        # --- GP 模型配置 ---
        self.observed_t = []      
        self.observed_val = []    
        
        # 使用 Matern 核通常比 RBF 更适合这种可能出现突变（拐点）的过程
        # nu=1.5 允许函数稍微粗糙一点，不如 RBF 那么平滑，适合找断崖
        kernel = ConstantKernel(1.0) * Matern(length_scale=50.0, nu=1.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

        # --- 状态控制 ---
        self.mode = 'binary_search'
        
        # 二分搜索状态
        self.low = min_noise
        self.high = max_noise
        self.safe_baseline = min_noise
        self.bs_history_scores = [] 

        # 停滞检测
        self.use_hv_trigger = use_hv_trigger
        self.centroid_history = deque(maxlen=cent_patience)
        self.centroid_dist_threshold = 1e-3
        self.cent_patience = cent_patience
        self.score = 0

    def _calculate_score(self, pop):
        """
        计算当前种群的'性能'。
        注意：如果你的目标是找'性能'随噪声上升的拐点，
        这个 score 必须反映那个'性能' (例如 Diversity * Validity)。
        如果这里只返回 Validity (通常随噪声下降)，逻辑需要反转。
        假设：你已经定义好了一个随噪声先升后降(或持平)的 score。
        """
        n_mols = len(pop)
        if n_mols == 0: return 0.0

        n_validity_rdkit = sum([float(mol.get('validity_rdkit', 0)) for mol in pop])
        n_mol_stable = sum([float(mol.get('mol_stable', 0)) for mol in pop])
        n_atm_stable = sum([float(mol.get('atm_stable', 0)) for mol in pop])

        score_atm = n_atm_stable / n_mols
        score_mol = n_mol_stable / n_mols
        score_val = n_validity_rdkit / n_mols

        return score_val * score_atm

    def update(self, Off, Pop, tracker, viz, obj_names, minimize=True, calHV=True, ref_info=None):
        """
        更新噪声水平并记录数据。
        ref_info: 若传入（如包含 ref_vina 的 info 字典），则开启对接模式的特殊归一化。
        """
        import torch
        current_score = self._calculate_score(Off)

        # [关键] 记录数据
        self.observed_t.append(self.curr_noise)
        self.observed_val.append(current_score)

        # ==================================================
        # 记录 HV (支持对接模式特化归一化)
        # ==================================================
        if calHV:
            if ref_info is not None:
                current_hv = viz.calculate_single_gen_hv_docking(Pop, obj_names, ref_info)

            else:
                # 常规无约束模式 / 简单小分子生成模式
                current_hv = viz.calculate_single_gen_hv(Pop, obj_names, minimize)

            if 'HV' not in tracker.metrics:
                tracker.metrics['HV'] = []
            tracker.metrics['HV'].append(current_hv)

        # 后续质心计算 (保持原样使用未归一化的原始坐标，反映真实的物理空间变化)
        pop_objs = []
        for item in Pop:
            vals = [item[obj].item() if torch.is_tensor(item[obj]) else item[obj] for obj in obj_names]
            pop_objs.append(vals)
        current_centroid = np.mean(np.array(pop_objs), axis=0)

        # 调度逻辑
        if self.mode == 'binary_search':
            self._step_binary_search(current_score)
        else:
            self._step_adaptive_gp_inflection(current_score, current_centroid)

        # 边界截断
        self.curr_noise = int(max(self.min_t, min(self.max_t, self.curr_noise)))
        self.score = current_score
        return self.curr_noise

    def _step_binary_search(self, score):
        """
        强制二分搜索阶段 (Coarse Exploration)
        """
        history_mean = np.mean(self.bs_history_scores) if self.bs_history_scores else score

        print(f"  [Scheduler-BS] Noise={self.curr_noise}, Score={score:.3f}, HistMean={history_mean:.3f}")

        # --- A. 停止条件：断崖式下降 ---
        if len(self.bs_history_scores) and score < (history_mean * self.drop_factor):
            print(f"  [Scheduler] Cliff Drop Detected ({score:.3f} < {self.drop_factor} * {history_mean:.3f}).")
            print(f"  [Scheduler] Switching to GP Adaptive Mode. Reverting to Safe Baseline: {self.safe_baseline}")
            self.mode = 'adaptive'
            self.curr_noise = self.safe_baseline
            return
        else:
            self.safe_baseline = self.curr_noise

        self.bs_history_scores.append(score)

        # 计算中点
        self.curr_noise = (self.low + self.high) // 2
        self.high = self.curr_noise
        
        # 如果二分区间已经很小，也强制切换到 GP 模式
        if (self.high - self.low) < 10:
             print(f"  [Scheduler] Binary Search Converged. Switching to GP Adaptive Mode.")
             self.mode = 'adaptive'

    def _step_adaptive_gp_inflection(self, score, current_centroid):
        """
        寻找性能拐点 (Inflection Point / Knee Point)
        目标：找到 max( Performance(t) - Cost(t) )
        """
        
        # 1. 拟合 GP
        X = np.array(self.observed_t).reshape(-1, 1)
        y = np.array(self.observed_val)
        
        try:
            self.gp.fit(X, y)
        except Exception:
            print("GP faild")
            self.curr_noise -= self.step_size
            return

        # 2. 全局预测
        t_candidates = np.arange(self.min_t, self.max_t + 1, 10).reshape(-1, 1)
        mu, sigma = self.gp.predict(t_candidates, return_std=True)

        t_norm = (t_candidates - self.min_t) / (self.max_t - self.min_t + 1e-6)

        beta = self.beta
        ucb = mu + beta * sigma

        acquisition_scores = ucb - (self.penalty_factor * t_norm.flatten())

        best_idx = np.argmax(acquisition_scores)
        target_t = t_candidates[best_idx][0]
        
        # 调试打印
        best_mu = mu[best_idx]
        best_sigma = sigma[best_idx]
        sigma_jitter = self.jittor_scale
        noisy_t = np.random.normal(loc=target_t, scale=sigma_jitter)
        next_t = int(np.clip(noisy_t, self.min_t, self.max_t))

        print(f"  [GP-Knee] Curr={self.curr_noise} Score={score:.3f} | Target={target_t}, Sampled={next_t} (Pred={best_mu:.2f}±{best_sigma:.2f})")

        self.curr_noise = int(next_t)

        # 4. 停滞检测 (原有逻辑 - 即使在最佳拐点，如果种群不动了，也要踢一脚)
        if self.use_hv_trigger:
            self._check_stagnation(current_centroid)

    def _check_stagnation(self, current_centroid):
        self.centroid_history.append(current_centroid)
        if len(self.centroid_history) == self.cent_patience:
            c_now = self.centroid_history[-1]
            c_prev1 = self.centroid_history[-2]
            c_prev2 = self.centroid_history[-3]
            dist1 = np.linalg.norm(c_now - c_prev1)
            dist2 = np.linalg.norm(c_prev1 - c_prev2)

            if dist1 < self.centroid_dist_threshold and dist2 < self.centroid_dist_threshold:
                boost = 200
                print(f"  [Scheduler] Stagnation! Boosting +{boost} to escape local optima.")
                self.curr_noise += boost
                self.centroid_history.clear()


def decouple_memory_references(data_list):
    """
    遍历列表，检查是否存在指向同一内存地址的元素。
    如果发现重复引用，则使用 deepcopy 将其强制转换为独立副本。

    此函数会直接修改原列表 (In-place)。
    """
    seen_ids = set()

    for i, item in enumerate(data_list):
        current_id = id(item)

        # 检查这个内存地址是否之前已经在这个列表中出现过
        if current_id in seen_ids:
            # 发现重复引用！执行深拷贝，断开内存关联
            # 注意：这里我们覆盖了列表中的当前位置
            new_item = copy.deepcopy(item)
            data_list[i] = new_item

            # (可选) 打印日志方便调试
            # print(f"索引 {i} 处的元素是重复引用，已执行 Deepcopy。")
        else:
            # 第一次遇到这个对象，记录它的内存地址
            seen_ids.add(current_id)

    return data_list


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


def get_fr_avgcon(Pop):
    if not Pop or len(Pop) == 0:
        return 0.0, 0.0  # <--- 加入这行保护

    current_con_vals = []
    for item in Pop:
        chem_con = max(0, item.get('Constraint', 0))
        struct_con = item.get('structure_Con', 0)
        if isinstance(struct_con, list):
            struct_con_sum = sum(struct_con)
        else:
            struct_con_sum = struct_con

        total_cv = chem_con + max(0, struct_con_sum)
        current_con_vals.append(total_cv)

    fr = np.mean([val == 0 for val in current_con_vals])
    avg_con = np.mean(current_con_vals)

    return float(fr), float(avg_con)




def get_population_duplicate_rate(Pop, dataset_info, NPops, usecon=True):
    """
    计算种群的【独一无二可行解占比】(Unique Feasible Rate)。
    只统计同时满足化学约束和结构约束的完美解，并基于 Canonical SMILES 进行 2D 拓扑精确去重。

    返回:
        unique_feasible_rate: 独一无二的完美可行解占总种群大小的比例 (0.0 ~ 1.0)
    """
    from rdkit import Chem
    import numpy as np

    valid_smiles_list = []

    for p in Pop:
        # =======================================================
        # 1. 严格判断：必须同时满足化学约束和结构约束
        # =======================================================

        chem_con = max(0, p.get('Constraint', 0))
        struct_con_sum = 0

        if usecon:
            s_con = p.get('structure_Con', 0)
            struct_con_sum = sum(s_con) if isinstance(s_con, list) else s_con
            struct_con_sum = max(0, struct_con_sum)

        total_cv = chem_con + struct_con_sum

        # 只要总约束 > 0 (即不可行)，直接跳过，绝不计入分子统计！
        if total_cv > 0:
            continue

        # =======================================================
        # 2. 提取或现场还原 SMILES (仅针对完美可行解)
        # =======================================================
        smi = p.get('valid_SMILES_rdkit', 'None')

        if smi == 'None' or smi == '':
            try:
                n_atoms = p['long']
                coords = p['x'][:n_atoms].detach().cpu().numpy()
                atom_types = [dataset_info['atom_decoder'][i] for i in
                              p['atom_type'][:n_atoms].detach().cpu().numpy()]

                # 假设之前已经写好了这个辅助函数
                mol = build_mol_from_coords_and_types(coords, atom_types)
                if mol is not None:
                    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            except Exception:
                continue

        # =======================================================
        # 3. 收集并标准化 SMILES
        # =======================================================
        if smi != 'None' and smi != '':
            try:
                # 强制转换为标准的 Canonical SMILES，消灭同物异名
                canon_smi = Chem.CanonSmiles(smi)
                valid_smiles_list.append(canon_smi)
            except:
                # 如果 RDKit 无法 Canonicalize，退而求其次用原字符串
                valid_smiles_list.append(smi)

    # 4. 统计与去重
    total_feasible = len(valid_smiles_list)
    if total_feasible == 0:
        # 如果连一个可行解都没有，自然为 0
        return 0.0

    # 利用 set 集合去重，得到独一无二的骨架数量
    unique_smiles = set(valid_smiles_list)
    unique_count = len(unique_smiles)

    # 返回：独一无二的可行解数量 / 种群设定大小
    # 如果输出是 1.0，说明 32 个槽位被 32 个完全不同的完美分子填满！
    unique_feasible_rate = unique_count / NPops

    return float(unique_feasible_rate)

