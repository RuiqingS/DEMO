from pymatgen.core import Lattice, Structure
import torch
import matplotlib.pyplot as plt
import matplotlib
# 【极其关键的修复】：强制使用无头渲染引擎，彻底杜绝 WSL/远程服务器卡死！
matplotlib.use("Agg")
import os
import gc
import sys
from pathlib import Path
from contextlib import contextmanager
from mattergen.generator import CrystalGenerator
from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo

_generator_cache: dict[str, CrystalGenerator] = {}


def _get_cached_generator(model_path: str) -> CrystalGenerator:
    """
    Reuse one CrystalGenerator per model path to avoid reloading model
    across multiple runs/attempts in the same Python process.
    """
    cache_key = str(Path(model_path).resolve())
    cached = _generator_cache.get(cache_key)
    if cached is not None:
        return cached

    checkpoint_info = MatterGenCheckpointInfo(model_path=Path(cache_key))
    generator = CrystalGenerator(checkpoint_info=checkpoint_info, record_trajectories=False)
    _generator_cache[cache_key] = generator
    return generator


@contextmanager
def _suppress_output():
    """
    Silence noisy third-party prints during model generation.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def dict_to_cif(crystal_dict: dict, file_path: str):
    """
    Saves a crystal dictionary to a CIF file.

    Args:
        crystal_dict: A dictionary containing the crystal structure.
        file_path: The path to save the CIF file.
    """
    lattice = Lattice(crystal_dict["cell"].squeeze(0).cpu().numpy())
    species_tensor = crystal_dict["atomic_numbers"].long().cpu()
    valid_species = torch.where(
        (species_tensor >= 1) & (species_tensor <= 118),
        species_tensor,
        torch.ones_like(species_tensor),
    )
    if not torch.equal(species_tensor, valid_species):
        print(
            f"Warning: Found invalid atomic numbers while writing {file_path}; "
            "replacing them with H (Z=1)."
        )
    species = valid_species.numpy()
    coords = crystal_dict["pos"].cpu().numpy()

    structure = Structure(lattice, species, coords, coords_are_cartesian=False)
    structure.to(fmt="cif", filename=file_path)


def cif_to_dict(cif_path: str) -> dict:
    """
    Reads a CIF file and converts it to a dictionary in the mattergen format.

    Args:
        cif_path: Path to the CIF file.

    Returns:
        A dictionary containing the crystal structure in mattergen format.
    """
    structure = Structure.from_file(cif_path)

    pos = torch.tensor(structure.frac_coords, dtype=torch.float32)
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    atomic_numbers = torch.tensor([site.specie.number for site in structure.sites], dtype=torch.long)
    num_nodes = len(structure.sites)

    return {
        "pos": pos,
        "cell": cell,
        "atomic_numbers": atomic_numbers,
        "num_nodes": num_nodes,
    }

# ================= 辅助函数 =================
def dict_to_structure(data_dict: dict) -> Structure:
    """
    将 mattergen 格式的字典转换回 pymatgen 的 Structure 对象。
    【优化】：具有缓存机制，第一次转换后会将其存入字典的 "structure_obj" 键中。
    后续所有函数再次调用本函数时，耗时将几乎为 0。
    """
    # 如果字典里已经缓存了这个结构，直接读取返回
    if "structure_obj" in data_dict:
        return data_dict["structure_obj"]

    # 如果没有缓存，则执行张量到结构的转换
    lattice = data_dict["cell"].squeeze(0).cpu().numpy()
    coords = data_dict["pos"].cpu().numpy()
    species = data_dict["atomic_numbers"].cpu().numpy()

    struct = Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=False)

    # 【关键步】存入字典，供后续函数复用
    data_dict["structure_obj"] = struct
    return struct


def structure_to_dict(structure) -> dict:
    """
    Convert a pymatgen Structure object to mattergen-style dict.
    """
    pos = torch.tensor([site.frac_coords for site in structure.sites], dtype=torch.float32)
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    atomic_numbers = torch.tensor([site.specie.number for site in structure.sites], dtype=torch.long)
    return {
        "pos": pos,
        "cell": cell,
        "atomic_numbers": atomic_numbers,
        "num_nodes": len(structure.sites),
    }


def generate_model_initial_pool(
    need_count: int,
    model_out_dir: str,
    run_seed: int,
    run_idx: int,
    constraints: dict,
    model_path: str,
    model_init_batch_size: int,
    model_init_max_attempts: int,
    extract_properties_fn,
    evaluate_constraints_fn,
) -> list[dict]:
    """
    Randomly sample crystals from MatterGen, then keep only those
    satisfying the provided constraints.
    """
    if need_count <= 0:
        return []

    generator = _get_cached_generator(model_path)
    accepted = []
    attempt = 0
    sample_idx = 0
    os.makedirs(model_out_dir, exist_ok=True)

    torch.manual_seed(run_seed)
    while len(accepted) < need_count and attempt < model_init_max_attempts:
        attempt += 1
        
        sampled_structures = generator.generate(
            batch_size=model_init_batch_size,
            num_batches=1,
            output_dir=model_out_dir,
        )
        for struct in sampled_structures:
            crystal = structure_to_dict(struct)
            crystal = extract_properties_fn(crystal)
            crystal = evaluate_constraints_fn(crystal, constraints)
            # if not crystal.get("is_valid", False):
            #     continue

            sample_name = f"init_model_run{run_idx:03d}_{sample_idx:04d}.cif"
            sample_path = os.path.join(model_out_dir, sample_name)
            dict_to_cif(crystal, sample_path)
            crystal["name"] = sample_name
            crystal["path"] = sample_path
            crystal["meta"] = {"type": "initial_model_generated", "parent_1": "None", "parent_2": "None"}
            accepted.append(crystal)
            sample_idx += 1
            if len(accepted) >= need_count:
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(accepted) < need_count:
        print(
            f"[Init Warning] Model init filled {len(accepted)}/{need_count} "
            f"after {attempt} attempts."
        )
    return accepted


def collect_objective_snapshot(
    population: list[dict],
    generation: int,
    objective_x: str,
    objective_y: str,
) -> dict:
    """
    采集当前代在目标空间的分布快照。
    """
    xs = []
    ys = []
    valids = []
    for cand in population:
        x_val = cand.get(objective_x)
        y_val = cand.get(objective_y)
        if x_val is None or y_val is None:
            continue
        xs.append(float(x_val))
        ys.append(float(y_val))
        valids.append(bool(cand.get("is_valid", False)))
    return {"generation": generation, "x": xs, "y": ys, "is_valid": valids}


import os
import matplotlib.pyplot as plt

POP_MARKERS = {
    "Pop_C (Feasible_Elite)": "s",  # 方块 (象征精英基石)
    "Pop_B (Infeasible_Guide)": "^",  # 上三角 (象征向上攀爬)
    "Pop_A (Diverse_Explore)": "D",  # 菱形 (象征探索多样)
    "Main_Pop": "o"  # 默认圆形
}


def plot_multimodal_evolution_trajectory(
        sampled_snapshots: list[dict],
        objective_x: str,
        objective_y: str,
        save_dir: str,
):
    if not sampled_snapshots: return
    os.makedirs(save_dir, exist_ok=True)

    pop_types = set()
    for snap in sampled_snapshots:
        for ind in snap["population"]:
            pop_types.add(ind.get("pop_type", "Main_Pop"))

    pop_types = list(pop_types)
    pop_types.append("Combined_All")

    for p_type in pop_types:
        fig, ax = plt.subplots(figsize=(9, 7))
        n = max(len(sampled_snapshots), 1)
        cmap = plt.get_cmap("viridis")

        centroid_x, centroid_y = [], []

        for idx, snap in enumerate(sampled_snapshots):
            gen = snap["generation"]
            pop = snap["population"]
            color = cmap(idx / max(n - 1, 1))

            # 为了在全家福中画出不同形状，按种群类别聚类画图
            for p_tag, marker_shape in POP_MARKERS.items():
                if p_type != "Combined_All" and p_type != p_tag:
                    continue  # 单种群视图，跳过不相关的

                sub_pop = [ind for ind in pop if ind.get("pop_type", "Main_Pop") == p_tag]
                if not sub_pop: continue

                valid_data = [(ind.get(objective_x), ind.get(objective_y), ind.get("is_valid", False))
                              for ind in sub_pop if isinstance(ind.get(objective_x), (int, float))]

                if not valid_data: continue

                xs = [d[0] for d in valid_data]
                ys = [d[1] for d in valid_data]
                valids = [d[2] for d in valid_data]

                inv_x, inv_y = [x for x, v in zip(xs, valids) if not v], [y for y, v in zip(ys, valids) if not v]
                val_x, val_y = [x for x, v in zip(xs, valids) if v], [y for y, v in zip(ys, valids) if v]

                if inv_x:
                    ax.scatter(inv_x, inv_y, s=45, marker=marker_shape, facecolors="none", edgecolors=[color],
                               alpha=0.8)
                if val_x:
                    ax.scatter(val_x, val_y, s=45, marker=marker_shape, facecolors=[color], edgecolors=[color],
                               alpha=0.9)

            # 每代的质心依然根据整体（在这个视图下的）来算
            all_xs = [ind.get(objective_x) for ind in pop if isinstance(ind.get(objective_x), (int, float))]
            all_ys = [ind.get(objective_y) for ind in pop if isinstance(ind.get(objective_y), (int, float))]
            if all_xs:
                cx, cy = sum(all_xs) / len(all_xs), sum(all_ys) / len(all_ys)
                centroid_x.append(cx)
                centroid_y.append(cy)
                ax.text(cx, cy, f"G{gen}", fontsize=8, color=color, fontweight='bold')

        if centroid_x:
            ax.plot(centroid_x, centroid_y, "--", color="black", linewidth=1.2, alpha=0.6, label="Centroid Trajectory")

        ax.set_xlabel(objective_x, fontsize=12)
        ax.set_ylabel(objective_y, fontsize=12)
        ax.set_title(f"Evolution Trajectory: {p_type}", fontsize=14, pad=15)
        ax.grid(alpha=0.3, linestyle="--")

        # 动态图例构建 (只展示当前图里有的形状)
        custom_lines = [plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="Centroid")]
        for p_tag, marker_shape in POP_MARKERS.items():
            if p_type == "Combined_All" or p_type == p_tag:
                custom_lines.append(
                    plt.Line2D([], [], marker=marker_shape, color="w", markerfacecolor="gray", markeredgecolor="black",
                               label=p_tag.split(' ')[0]))

        custom_lines.append(
            plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="black",
                       label="Infeasible"))
        custom_lines.append(
            plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black",
                       label="Feasible"))

        ax.legend(handles=custom_lines, loc="best", fontsize=9, framealpha=0.9)

        fig.tight_layout()
        safe_name = p_type.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(os.path.join(save_dir, f"trajectory_{safe_name}.png"), dpi=250)
        plt.close(fig)


def plot_final_elite_pareto(
        final_elite_records: list[dict],
        objective_x: str,
        objective_y: str,
        save_path: str,
):
    """
    终极精英 Pareto 分布图。
    修复了键名读取的 BUG，并补全了智能图例压缩的完整代码。
    """
    if not final_elite_records:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # 【修复 1】：直接使用 FNDS 原生的 pareto_rank 键
    ranks = [r.get("pareto_rank") for r in final_elite_records if r.get("pareto_rank") is not None]
    if not ranks:
        plt.close(fig)
        return

    max_rank = max(ranks)
    cmap = plt.get_cmap("tab10") if max_rank <= 10 else plt.get_cmap("rainbow")

    for current_rank in range(1, max_rank + 1):
        # 筛选当前层的个体
        layer_data = [r for r in final_elite_records if r.get("pareto_rank") == current_rank]
        if not layer_data:
            continue

        # 【修复 2】：因为传入的是未经格式化的原始数据，这里可以安全读到 ehull_eV
        xs = [r.get(objective_x) for r in layer_data if isinstance(r.get(objective_x), (int, float))]
        ys = [r.get(objective_y) for r in layer_data if isinstance(r.get(objective_y), (int, float))]

        if not xs:
            continue

        color = cmap(current_rank - 1) if max_rank <= 10 else cmap((current_rank - 1) / max(max_rank - 1, 1))

        # 只对前 5 层显示在图例中，防止图例挤爆
        label = f"Rank {current_rank}" if current_rank <= 5 else "_nolegend_"
        ax.scatter(xs, ys, label=label, color=color, s=80, edgecolors='black', linewidths=0.8, alpha=0.9)

        # 专享特效：为 Rank 1 画出帕累托前沿曲线！
        if current_rank == 1:
            pts = sorted(zip(xs, ys))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle='-', color=color, linewidth=2.5, alpha=0.8)

    # 坐标轴美化 (可以使用一些替换逻辑让坐标轴更好看)
    display_x = "Energy Above Hull (eV/atom)" if objective_x == "ehull_eV" else objective_x
    display_y = "In-plane Dielectric Constant" if objective_y == "dielectric_epsx" else objective_y

    ax.set_xlabel(display_x, fontsize=12, fontweight='bold')
    ax.set_ylabel(display_y, fontsize=12, fontweight='bold')
    ax.set_title("Final Elite Pareto Fronts (All Qualified Crystals)", fontsize=15, pad=15, fontweight='bold')
    ax.grid(alpha=0.25, linestyle="-.")

    # 【补全上次截断的代码】：智能图例压缩
    handles, labels = ax.get_legend_handles_labels()
    if max_rank > 5:
        # 添加一个 "Other Ranks" 的代表性黑色小点
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8))
        labels.append("Rank > 5 (Hidden)")

    if handles:
        ax.legend(handles, labels, loc="best", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)  # 300 dpi 高清出图
    plt.close(fig)


import torch
import numpy as np

# ================= 全自动排版字典 =================
# 如果字典里出现了这些键，它会自动将其转为高大上的“学术名+单位”并限制小数位数
# 如果未来你加了新的属性(不在这个表里)，程序也不会报错，会自动用原变量名打印！
_PRINT_MAPPING = {
    # --- 新增的化学式和空间群 ---
    "formula": ("Formula", ""),
    "spacegroup": ("SpaceGroup", ""),
    "num_elements": ("K_nary", ""),

    # --- 原有的属性 ---
    "formation_energy_eV": ("E_form(eV/atom)", ".3f"),
    "exfoliation_energy_meV": ("Exf_E(meV/atom)", ".2f"),
    "chgnet_energy": ("E_tot(eV/atom)", ".4f"),
    "ehull_eV": ("E_hull(eV/atom)", ".3f"),
    "cl_score": ("CLscore(JACS)", ".3f"),
    "bandgap_eV": ("Bandgap(eV)", ".2f"),
    "dielectric_epsx": ("Eps_X", ".2f"),
    "total_magnetization": ("Mag(uB)", ".2f"),
    "actual_vacuum_gap": ("Vacuum(A)", ".4f"),
    "material_thickness": ("Thickness(A)", ".2f"),
    "in_plane_anisotropy_ratio": ("Aniso_Ratio", ".3f"),
    "f_max_eV_A": ("F_max(eV/A)", ".3f"),
    "s_max_GPa": ("S_max(GPa)", ".3f"),
    "total_violation": ("Violation", ".4f"),
    "fitness": ("Fitness", ".4f"),
    "pareto_rank": ("Pareto_Rank", "")
}

_DEFAULT_EXCLUDE = {
    "pos", "cell", "atomic_numbers", "num_nodes", "structure_obj",
    "meta", "path", "name", "dict", "dominated_set", "domination_count",
    "raw_forces", "raw_stress", "sim_to_C", "_backup_viol", "crowding_distance", "spea2_fitness"
}


def print_crystal_properties(name: str, crystal_dict: dict, exclude_keys: set = None):
    """
    全自动智能排版的打印函数。每行最多放 4 个属性，宛如精美的实验报表。
    """
    excludes = exclude_keys or _DEFAULT_EXCLUDE
    print(f"[{name}]")

    items = []
    for k, v in crystal_dict.items():
        # 自动屏蔽被指定的键，以及底层的 Tensors / Numpy 矩阵
        if k in excludes or hasattr(v, "cpu") or isinstance(v, np.ndarray):
            continue

        # 异常值检测
        if v is None or v == -1.0 or v == 999.0 or v == float('inf'):
            val_str = "unevaluated"
        else:
            # 格式化数值
            if k in _PRINT_MAPPING:
                fmt = _PRINT_MAPPING[k][1]
                val_str = format(float(v), fmt) if fmt else str(v)
            else:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)

        # 映射显示名称
        display_name = _PRINT_MAPPING.get(k, (k, ""))[0]
        items.append(f"{display_name}={val_str}")

    # 优雅分块打印，每 4 个属性折行一次
    chunk_size = 4
    for i in range(0, len(items), chunk_size):
        print("  - " + ", ".join(items[i:i + chunk_size]))
    print("-" * 75)


def auto_extract_record(crystal_dict: dict, exclude_keys: set = None) -> dict:
    """
    为 CSV / JSON 报告全自动提取扁平化数据。
    【优化】：现在 CSV 的表头也会自动使用 _PRINT_MAPPING 中的学术名了！
    """
    excludes = exclude_keys or _DEFAULT_EXCLUDE
    record = {}

    for k, v in crystal_dict.items():
        if k in excludes or hasattr(v, "cpu") or isinstance(v, np.ndarray):
            continue

        # 映射表头名字（如果有对应的映射名，则使用映射名，否则使用原键名）
        display_name = _PRINT_MAPPING.get(k, (k, ""))[0]

        if v is None or v == -1.0 or v == 999.0 or v == float('inf'):
            record[display_name] = "unevaluated"
        elif isinstance(v, float):
            # 格式化，优先按 Mapping 的要求处理
            fmt = _PRINT_MAPPING.get(k, ("", ""))[1]
            if fmt:
                record[display_name] = round(v, int(fmt[1]))  # 提取精度如 .3f 里的 3
            else:
                record[display_name] = round(v, 4)
        else:
            record[display_name] = v

    return record


