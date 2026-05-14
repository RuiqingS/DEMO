import torch
import random
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from typing import Any
from omegaconf import OmegaConf
from hydra.utils import instantiate
from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.utils.globals import get_device, DEFAULT_SAMPLING_CONFIG_PATH, MAX_ATOMIC_NUM
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    NumAtomsVarianceAdjustedWrappedVESDE,
)
import os

# Global cache for model to avoid reloading
_model_cache = {}
_corruption_cache: dict[str, dict[str, Any]] = {}


def to_noise_input(crystal: dict) -> dict:
    """
    Build a minimal crystal dict for add_noise to avoid copying
    large cached fields (e.g., structure objects / derived metadata).
    """
    return {
        "pos": crystal["pos"],
        "cell": crystal["cell"],
        "atomic_numbers": crystal["atomic_numbers"],
        "num_nodes": crystal["num_nodes"],
    }


def select_parents(
    parent_source: list[dict],
    mode: str = "random",
    tournament_k: int = 3,
) -> tuple[dict, dict]:
    """
    Select two distinct parents from parent_source.
    mode:
    - random: uniform random sampling
    - tournament: k-way tournament by fitness
    """
    if len(parent_source) < 2:
        raise ValueError("select_parents requires at least 2 candidates.")

    if mode == "random":
        return tuple(random.sample(parent_source, 2))

    if mode != "tournament":
        raise ValueError(f"Unknown parent selection mode: {mode}")

    k = max(2, min(tournament_k, len(parent_source)))

    def _pick_one(exclude_id: int | None = None) -> dict:
        candidates = [c for c in parent_source if id(c) != exclude_id]
        sampled = random.sample(candidates, k if len(candidates) >= k else len(candidates))
        return max(sampled, key=lambda c: float(c.get("fitness", -1e9)))

    p1 = _pick_one()
    p2 = _pick_one(exclude_id=id(p1))
    return p1, p2


def add_noise(crystal_dict: dict, t: float, model_path_or_name: str) -> dict:
    """
    Adds noise to a crystal dictionary.

    Args:
        crystal_dict: A dictionary containing the crystal structure in mattergen format.
        t: The noise level, a float between 0 and 1.
        model_path_or_name: model directory used to load training-time corruption config.

    Returns:
        A dictionary containing the noisy crystal structure and the noise level 't'.
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Noise level t must be in [0, 1], got {t}.")

    corruption_bundle = load_corruptions_from_model_config(model_path_or_name)
    lattice_sde = corruption_bundle["lattice_sde"]
    pos_sde = corruption_bundle["pos_sde"]
    atomic_corruption = corruption_bundle["atomic_corruption"]

    device = get_device()
    num_nodes = int(crystal_dict["num_nodes"])
    chem_graph = ChemGraph(
        pos=crystal_dict["pos"].to(device),
        cell=crystal_dict["cell"].to(device),
        atomic_numbers=crystal_dict["atomic_numbers"].long().to(device),
        num_atoms=torch.tensor(num_nodes, device=device),
        num_nodes=torch.tensor(num_nodes, device=device),
    )
    batched_graph = collate([chem_graph]).to(device)

    # Ensure t is a 1D tensor for batch compatibility
    t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
    pos_batch_idx = batched_graph.get_batch_idx("pos")
    atomic_batch_idx = batched_graph.get_batch_idx("atomic_numbers")

    # pos: wrapped VESDE with std scaled by num_atoms
    noisy_pos = pos_sde.sample_marginal(
        batched_graph["pos"], t_tensor, batch_idx=pos_batch_idx, batch=batched_graph
    )
    # cell: lattice VPSDE (limit mean/var conditioned by num_atoms)
    noisy_cell = lattice_sde.sample_marginal(batched_graph["cell"], t_tensor, batch=batched_graph)
    # atom type: D3PM
    noisy_atomic_numbers = atomic_corruption.sample_marginal(
        batched_graph["atomic_numbers"].long(),
        t_tensor,
        batch_idx=atomic_batch_idx,
        batch=batched_graph,
    )

    return {
        "pos": noisy_pos,
        "cell": noisy_cell,
        "atomic_numbers": noisy_atomic_numbers,
        "num_nodes": crystal_dict["num_nodes"],
        "t": t,  # Include the noise level in the output
    }


def align_lattice(L1: torch.Tensor, L2: torch.Tensor) -> torch.Tensor:
    """
    使用 Kabsch 算法 (Orthogonal Procrustes) 将 L2 旋转对齐到 L1。
    寻找最优旋转矩阵 R，使得 ||L1 - R @ L2||_F 最小化。

    Args:
        L1, L2: shape 为 [3, 3] 的晶格张量
    Returns:
        对齐后的 L2 晶格张量
    """
    # 计算协方差矩阵 H
    H = L2 @ L1.transpose(-1, -2)

    # 奇异值分解 SVD
    # PyTorch 的 svd 返回 U, S, Vh (其中 Vh 是 V 的转置)
    U, S, Vh = torch.linalg.svd(H)

    # 计算初始旋转矩阵 R = V @ U^T = Vh^T @ U^T
    R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)

    # 检查是否发生了镜像翻转 (行列式为 -1)
    # 晶格在物理三维空间中只能旋转，不能镜像翻转
    if torch.linalg.det(R) < 0:
        Vh[-1, :] *= -1  # 翻转最后一个主轴的方向
        R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)

    return R @ L2



def crossover(parent1: dict, parent2: dict, limit_density: float | None = None) -> dict:
    """
    Performs a robust crossover operation between two noisy crystal dictionaries.
    Includes optimal lattice rotation alignment and exact volume/density scaling.
    """
    t1 = float(parent1.get("t"))
    t2 = float(parent2.get("t"))
    if abs(t1 - t2) > 1e-8:
        raise ValueError("Crossover requires parents with the same noise level t.")

    device = parent1["pos"].device

    # 1. 提取基础信息
    n1 = int(parent1["num_nodes"])
    n2 = int(parent2["num_nodes"])
    cell1 = parent1["cell"]
    cell2 = parent2["cell"]

    # --- 【核心改进 1：最优旋转对齐】 ---
    # 在求平均之前，先把父代 2 的晶格转到和父代 1 方向一致
    cell2_aligned = align_lattice(cell1, cell2)

    # 2. 决定子代原子数量
    max_child_atoms = min(19, n1 + n2)
    if max_child_atoms < 1:
        raise ValueError("At least one atom is required across both parents for crossover.")

    min_child_atoms = 4 if max_child_atoms >= 4 else 1
    child_num_atoms = int(
        torch.randint(min_child_atoms, max_child_atoms + 1, (1,), device=device).item()
    )

    # 3. 从合并池中随机抽取原子
    parent_pos_pool = torch.cat([parent1["pos"][:n1], parent2["pos"][:n2]], dim=0)
    parent_atomic_pool = torch.cat(
        [parent1["atomic_numbers"][:n1], parent2["atomic_numbers"][:n2]], dim=0
    )

    chosen_idx = torch.randperm(parent_pos_pool.shape[0], device=device)[:child_num_atoms]

    child_pos = parent_pos_pool[chosen_idx]
    child_atomic_numbers = parent_atomic_pool[chosen_idx]

    # --- 【核心改进 2：基于实际抽样的体积密度守恒】 ---
    # 精确统计：这次抽卡，到底抽到了多少个父代1的原子，多少个父代2的原子？
    k1 = (chosen_idx < n1).sum().item()
    k2 = child_num_atoms - k1

    # 计算父代的单原子平均体积
    vol1 = torch.abs(torch.det(cell1))
    vol2 = torch.abs(torch.det(cell2))
    atomic_vol1 = vol1 / max(n1, 1)
    atomic_vol2 = vol2 / max(n2, 1)

    # 子代应该拥有的完美物理体积（绝对守恒）
    target_vol = k1 * atomic_vol1 + k2 * atomic_vol2

    # 对齐后求平均，得到原始的融合晶格（保留了形状特征）
    child_cell_raw = (cell1 + cell2_aligned) / 2.0
    raw_vol = torch.abs(torch.det(child_cell_raw))

    # 如果给定训练中的 limit_density，直接使用与训练一致的对角晶格目标长度
    if limit_density is not None and limit_density > 0:
        target_len = (child_num_atoms / limit_density) ** (1.0 / 3.0)
        child_cell = (
            torch.eye(3, device=device, dtype=child_cell_raw.dtype).unsqueeze(0) * target_len
        )
    # 根据目标体积进行等比例缩放修正
    elif raw_vol < 1e-5:
        # 极端情况防爆保护：如果平均后体积依然意外坍缩，回退到标准正方体
        child_cell = torch.eye(3, device=device, dtype=child_cell_raw.dtype).unsqueeze(0) * (
            target_vol ** (1 / 3)
        )
    else:
        scale_factor = (target_vol / raw_vol) ** (1 / 3)
        child_cell = child_cell_raw * scale_factor

    return {
        "pos": child_pos,
        "cell": child_cell,
        "atomic_numbers": child_atomic_numbers,
        "num_nodes": child_num_atoms,
        "t": t1,  # Propagate the noise level
    }



def _load_model(model_path_or_name: str):
    if model_path_or_name in _model_cache:
        return _model_cache[model_path_or_name]

    print(f"Loading model: {model_path_or_name}...")
    if os.path.isdir(model_path_or_name):
        checkpoint_info = MatterGenCheckpointInfo(model_path=Path(model_path_or_name).resolve())
    else:
        checkpoint_info = MatterGenCheckpointInfo.from_local(model_path_or_name)

    model = load_model_diffusion(checkpoint_info)
    model = model.to(get_device())
    model.eval()
    _model_cache[model_path_or_name] = model
    print("Model loaded and cached.")
    return model


def load_corruptions_from_model_config(model_path_or_name: str) -> dict[str, Any]:
    if model_path_or_name in _corruption_cache:
        return _corruption_cache[model_path_or_name]

    config_path = Path(model_path_or_name) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find model config: {config_path}")

    model_cfg = OmegaConf.load(config_path)
    corruption_cfg = model_cfg.lightning_module.diffusion_module.corruption

    lattice_cfg = corruption_cfg.sdes.cell.vpsde_config
    lattice_sde = LatticeVPSDE.from_vpsde_config(lattice_cfg)

    pos_cfg = corruption_cfg.sdes.pos
    pos_sde = NumAtomsVarianceAdjustedWrappedVESDE(
        wrapping_boundary=float(pos_cfg.wrapping_boundary),
        sigma_max=float(pos_cfg.sigma_max),
        limit_info_key=str(pos_cfg.limit_info_key),
    )

    atomic_cfg = corruption_cfg.discrete_corruptions.atomic_numbers
    schedule_cfg = atomic_cfg.d3pm.schedule
    schedule = create_discrete_diffusion_schedule(
        kind=str(schedule_cfg.kind),
        num_steps=int(schedule_cfg.num_steps),
    )
    d3pm = MaskDiffusion(dim=int(atomic_cfg.d3pm.dim), schedule=schedule)
    atomic_corruption = D3PMCorruption(d3pm=d3pm, offset=int(atomic_cfg.offset))

    corruption_bundle = {
        "lattice_sde": lattice_sde,
        "pos_sde": pos_sde,
        "atomic_corruption": atomic_corruption,
        "limit_density": float(lattice_cfg.limit_density),
    }
    _corruption_cache[model_path_or_name] = corruption_bundle
    return corruption_bundle

def denoise(
    noisy_crystal_dict: dict,
    model_path_or_name: str,
    mask: dict = None,
) -> dict:
    """
    Denoises a noisy crystal dictionary using a pre-trained model, starting from an intermediate timestep t.

    Args:
        noisy_crystal_dict: A dictionary containing the noisy crystal structure and a 't' key for the noise level.
        model_path_or_name: The name or path of the pre-trained model to use.
        mask: Optional mask for inpainting.

    Returns:
        A dictionary containing the denoised crystal structure.
    """
    model = _load_model(model_path_or_name)

    start_t = noisy_crystal_dict.pop('t')

    # Instantiate sampler with the correct starting time (max_t)
    sampling_config = OmegaConf.load(DEFAULT_SAMPLING_CONFIG_PATH / "default.yaml")
    sampler_partial = instantiate(sampling_config.sampler_partial)
    sampler = sampler_partial(pl_module=model, max_t=start_t)
    sampler.N = int(sampling_config.sampler_partial.N * start_t)

    atomic_numbers = noisy_crystal_dict["atomic_numbers"].long().clone()
    invalid_atomic_numbers = (atomic_numbers < 1) | (atomic_numbers > MAX_ATOMIC_NUM)
    if invalid_atomic_numbers.any():
        valid_atomic_numbers = atomic_numbers[~invalid_atomic_numbers]
        if valid_atomic_numbers.numel() == 0:
            atomic_numbers[invalid_atomic_numbers] = 1
        else:
            random_idx = torch.randint(
                low=0,
                high=valid_atomic_numbers.numel(),
                size=(int(invalid_atomic_numbers.sum().item()),),
                device=atomic_numbers.device,
            )
            atomic_numbers[invalid_atomic_numbers] = valid_atomic_numbers[random_idx]

    chem_graph = ChemGraph(
        pos=noisy_crystal_dict["pos"],
        cell=noisy_crystal_dict["cell"],
        atomic_numbers=atomic_numbers,
        num_atoms=torch.tensor(noisy_crystal_dict["num_nodes"]),
        num_nodes=torch.tensor(noisy_crystal_dict["num_nodes"]),
    )

    batched_data = collate([chem_graph])
    batched_data = batched_data.to(get_device())

    with torch.no_grad():
        _, final_mean, _ = sampler._denoise(batch=batched_data, mask=mask or {}, record=False)

    result_list = final_mean.to_data_list()
    result_dict = result_list[0]

    return {
        "pos": result_dict["pos"],
        "cell": result_dict["cell"],
        "atomic_numbers": result_dict["atomic_numbers"],
        "num_nodes": result_dict["num_nodes"].item(),
    }


def _sanitize_atomic_numbers_for_model(atomic_numbers: torch.Tensor) -> torch.Tensor:
    atomic_numbers = atomic_numbers.long().clone()
    invalid_atomic_numbers = (atomic_numbers < 1) | (atomic_numbers > MAX_ATOMIC_NUM)
    if invalid_atomic_numbers.any():
        valid_atomic_numbers = atomic_numbers[~invalid_atomic_numbers]
        if valid_atomic_numbers.numel() == 0:
            atomic_numbers[invalid_atomic_numbers] = 1
        else:
            random_idx = torch.randint(
                low=0,
                high=valid_atomic_numbers.numel(),
                size=(int(invalid_atomic_numbers.sum().item()),),
                device=atomic_numbers.device,
            )
            atomic_numbers[invalid_atomic_numbers] = valid_atomic_numbers[random_idx]
    return atomic_numbers


def denoise_batch(
    noisy_crystal_dicts: list[dict],
    model_path_or_name: str,
    masks: list[dict] | None = None,
) -> list[dict]:
    """
    Denoises a batch of noisy crystals in one sampler run.
    """
    if len(noisy_crystal_dicts) == 0:
        return []
    if masks is None:
        masks = [{} for _ in noisy_crystal_dicts]
    if len(masks) != len(noisy_crystal_dicts):
        raise ValueError("masks must have the same length as noisy_crystal_dicts.")

    start_ts = [float(d["t"]) for d in noisy_crystal_dicts]
    first_t = start_ts[0]
    if any(abs(t - first_t) > 1e-8 for t in start_ts):
        raise ValueError("All noisy crystals in denoise_batch must share the same t.")

    model = _load_model(model_path_or_name)
    sampling_config = OmegaConf.load(DEFAULT_SAMPLING_CONFIG_PATH / "default.yaml")
    sampler_partial = instantiate(sampling_config.sampler_partial)
    sampler = sampler_partial(pl_module=model, max_t=first_t)
    sampler.N = int(sampling_config.sampler_partial.N * first_t)

    chem_graphs = []
    for noisy_crystal_dict in noisy_crystal_dicts:
        atomic_numbers = _sanitize_atomic_numbers_for_model(noisy_crystal_dict["atomic_numbers"])
        chem_graphs.append(
            ChemGraph(
                pos=noisy_crystal_dict["pos"],
                cell=noisy_crystal_dict["cell"],
                atomic_numbers=atomic_numbers,
                num_atoms=torch.tensor(noisy_crystal_dict["num_nodes"]),
                num_nodes=torch.tensor(noisy_crystal_dict["num_nodes"]),
            )
        )

    batched_data = collate(chem_graphs)
    batched_data = batched_data.to(get_device())
    with torch.no_grad():
        _, final_mean, _ = sampler._denoise(batch=batched_data, mask={}, record=False)

    results = []
    for result_dict in final_mean.to_data_list():
        results.append(
            {
                "pos": result_dict["pos"],
                "cell": result_dict["cell"],
                "atomic_numbers": result_dict["atomic_numbers"],
                "num_nodes": result_dict["num_nodes"].item(),
            }
        )
    return results
