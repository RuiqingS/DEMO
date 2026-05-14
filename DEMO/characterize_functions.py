from chgnet.model import CHGNet
import alignn.pretrained
from jarvis.core.atoms import Atoms
import os
import gc
import json
import shutil
import tempfile
import torch
import numpy as np
from torch.utils.data import DataLoader
from utilis import *
import alignn
from alignn.pretrained import get_figshare_model
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from alignn.models.alignn import ALIGNN
from jarvis.core.graphs import Graph
import xmlrpc.client
from alignn.pretrained import get_prediction

_model_path_cache = {}  # 内存级别的路径缓存
print("正在加载 CHGNet 模型...")
try:
    chgnet_model = CHGNet.load()
except Exception as e:
    print(f"CHGNet 加载失败: {e}")
try:
    jacs_proxy = xmlrpc.client.ServerProxy("http://localhost:8080/")
except Exception as e:
    print(f"无法连接到 JACS 微服务，请确保后台已运行 jacs_server.py: {e}")
# ================= 核心拦截补丁：修复 ALIGNN 狂写硬盘与重复加载 =================
_orig_get_figshare_model = alignn.pretrained.get_figshare_model
_ALIGNN_MODEL_CACHE = {}  # 内存级别的【模型对象】缓存
# ================= 安全隔离的 ALIGNN 推理引擎 =================
def _safe_alignn_predict(model_name: str, jarvis_atoms) -> float:
    """
    复用官方 get_prediction (官方已修复依赖问题)，但套上 Linux 内存盘沙盒，
    防止任何 DGL 图数据的临时文件写坏硬盘。
    """
    original_cwd = os.getcwd()
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(prefix="alignn_iso_", dir=base_dir)

    try:
        os.chdir(temp_dir)
        with suppress_output():
            # 这里的 get_prediction 内部会调用被我们 patch 过的函数
            # 瞬间从内存拿到模型，0 磁盘/加载开销！
            pred = get_prediction(model_name=model_name, atoms=jarvis_atoms)

        if isinstance(pred, (list, tuple, np.ndarray)):
            return float(pred[0])
        return float(pred)
    except Exception as e:
        raise e
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _patched_get_figshare_model(model_name):
    """
    在新版 ALIGNN 中，get_figshare_model 会直接返回加载好的 PyTorch 模型对象。
    我们拦截它，实现内存级单例常驻，速度提升数百倍！
    """
    if model_name in _ALIGNN_MODEL_CACHE:
        return _ALIGNN_MODEL_CACHE[model_name]

    print(f"\n[补丁拦截] 正在首次下载并加载模型: {model_name} (若卡住说明网络不通)...")
    model_obj = _orig_get_figshare_model(model_name)
    _ALIGNN_MODEL_CACHE[model_name] = model_obj
    return model_obj

alignn.pretrained.get_figshare_model = _patched_get_figshare_model

@contextmanager
def suppress_output():
    """强制屏蔽一切 print 输出的上下文管理器"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ================= 辅助函数 =================
def pmg_to_jarvis(struct) -> Atoms:
    return Atoms(
        lattice_mat=struct.lattice.matrix,
        elements=[str(site.specie) for site in struct],
        coords=struct.frac_coords,
        cartesian=False
    )


def reconstruct_crystal_geometry(struct: Structure, target_vacuum=None, target_vdw_gap=None):
    max_vacuums = []
    gap_centers = []
    for i in range(3):
        frac_coords = np.sort(struct.frac_coords[:, i])
        if len(frac_coords) <= 1:
            max_vacuums.append(struct.lattice.abc[i])
            gap_centers.append(0.5)
            continue

        diffs = np.diff(frac_coords)
        diffs = np.append(diffs, 1.0 + frac_coords[0] - frac_coords[-1])
        max_idx = np.argmax(diffs)
        max_vacuums.append(np.max(diffs) * struct.lattice.abc[i])

        if max_idx < len(frac_coords) - 1:
            center = (frac_coords[max_idx] + frac_coords[max_idx + 1]) / 2.0
        else:
            center = (frac_coords[-1] + 1.0 + frac_coords[0]) / 2.0
        gap_centers.append(center % 1.0)

    vacuum_axis = np.argmax(max_vacuums)
    actual_vacuum = max_vacuums[vacuum_axis]
    gap_center = gap_centers[vacuum_axis]

    struct_centered = struct.copy()
    shift_vec = [0.0, 0.0, 0.0]
    shift_vec[vacuum_axis] = -gap_center
    struct_centered.translate_sites(range(len(struct_centered)), shift_vec, to_unit_cell=True)

    cart_coords = struct_centered.cart_coords
    min_val = np.min(cart_coords[:, vacuum_axis])
    max_val = np.max(cart_coords[:, vacuum_axis])
    material_thickness = max_val - min_val

    if target_vacuum is not None:
        new_c = material_thickness + target_vacuum
    elif target_vdw_gap is not None:
        new_c = material_thickness + target_vdw_gap
    else:
        return struct_centered, actual_vacuum, material_thickness

    new_matrix = struct_centered.lattice.matrix.copy()
    new_matrix[vacuum_axis] *= (new_c / struct_centered.lattice.abc[vacuum_axis])

    new_cart_coords = cart_coords.copy()
    new_cart_coords[:, vacuum_axis] -= min_val
    new_cart_coords[:, vacuum_axis] += (new_c - material_thickness) / 2.0

    new_struct = Structure(lattice=new_matrix, species=struct_centered.species,
                           coords=new_cart_coords, coords_are_cartesian=True)
    return new_struct, actual_vacuum, material_thickness


def characterize_chgnet_basic(data_dict: dict) -> dict:
    """使用 CHGNet 提取磁矩 (m)、受力 (f) 和应力 (s)"""
    try:
        struct = dict_to_structure(data_dict)
        # 强制开启局部梯度，防止受力计算报错
        with torch.enable_grad():
            pred = chgnet_model.predict_structure(struct)

        # 1. 磁矩 (Magnetization)
        magmoms = pred.get('m', [])
        total_mag = float(sum(abs(m) for m in magmoms))

        # 2. 原子受力 (Forces)
        forces = np.array(pred.get('f', []))
        if forces.size > 0:
            f_max = float(np.max(np.linalg.norm(forces, axis=1)))
            raw_forces = forces.tolist()  # 正常情况下转为 list
        else:
            f_max = 999.0
            raw_forces = []

        # 3. 晶胞应力 (Stress)
        stress = np.array(pred.get('s', []))
        if stress.size > 0:
            s_max = float(np.max(np.abs(stress)))
            raw_stress = stress.tolist()  # 正常情况下转为 list
        else:
            s_max = 999.0
            raw_stress = []

    except Exception as e:
        # 当遇到“孤立原子 (isolated atoms)”等垃圾结构导致 CHGNet 崩溃时，
        # 我们会安全地落入这里，并给它判极高的 F_max 惩罚！
        print(f"CHGNet 力学计算报错 (该垃圾结构将被淘汰): {e}")
        total_mag = 0.0
        f_max = 999.0
        s_max = 999.0
        raw_forces = []
        raw_stress = []

    # 存入标量，方便查看和约束
    data_dict["total_magnetization"] = total_mag
    data_dict["has_magnetism"] = bool(total_mag > 0.5)
    data_dict["f_max_eV_A"] = f_max
    data_dict["s_max_GPa"] = s_max

    # 安全地存入完整的原始受力矩阵 (如果外部需要的话)
    data_dict["raw_forces"] = raw_forces
    data_dict["raw_stress"] = raw_stress

    return data_dict


def characterize_exfoliation_and_vacuum(data_dict: dict) -> dict:
    try:
        struct = dict_to_structure(data_dict)
        _, actual_vacuum, thickness = reconstruct_crystal_geometry(struct)

        # 【修复】：将所有的 CHGNet 调用包裹在 enable_grad 中
        with torch.enable_grad():
            if actual_vacuum >= 10.0:
                e_2d = float(chgnet_model.predict_structure(struct)['e'])
                struct_bulk, _, _ = reconstruct_crystal_geometry(struct, target_vdw_gap=3.2)
                e_bulk = float(chgnet_model.predict_structure(struct_bulk)['e'])
            else:
                e_bulk = float(chgnet_model.predict_structure(struct)['e'])
                struct_2d, _, _ = reconstruct_crystal_geometry(struct, target_vacuum=15.0)
                e_2d = float(chgnet_model.predict_structure(struct_2d)['e'])

        exf_energy = (e_2d - e_bulk) * 1000
    except Exception as e:
        print(f"剥离能报错: {e}")
        exf_energy = 9999.0
        actual_vacuum = 0.0
        thickness = 0.0

    data_dict["exfoliation_energy_meV"] = exf_energy
    data_dict["actual_vacuum_gap"] = actual_vacuum
    data_dict["material_thickness"] = thickness
    return data_dict


# ================= 基于安全引擎的四大高阶表征 =================

def characterize_formation_energy_alignn(data_dict: dict) -> dict:
    try:
        struct = dict_to_structure(data_dict)
        e_form = _safe_alignn_predict("jv_formation_energy_peratom_alignn", pmg_to_jarvis(struct))
        data_dict["formation_energy_eV"] = e_form
    except Exception as e:
        print(f"[E_form Error]: {e}")
        data_dict["formation_energy_eV"] = 999.0
    return data_dict


def characterize_bandgap(data_dict: dict) -> dict:
    try:
        struct = dict_to_structure(data_dict)
        bg = _safe_alignn_predict("jv_optb88vdw_bandgap_alignn", pmg_to_jarvis(struct))
        data_dict["bandgap_eV"] = max(0.0, bg)
    except Exception as e:
        print(f"[Bandgap Error]: {e}")
        data_dict["bandgap_eV"] = 999.0
    return data_dict


def characterize_dielectric_constant(data_dict: dict) -> dict:
    try:
        struct = dict_to_structure(data_dict)
        epsx = _safe_alignn_predict("jv_epsx_alignn", pmg_to_jarvis(struct))
        data_dict["dielectric_epsx"] = max(1.0, epsx)
    except Exception as e:
        print(f"[Dielectric Error]: {e}")
        data_dict["dielectric_epsx"] = None
    return data_dict


def characterize_synthesizability_ehull(data_dict: dict) -> dict:
    try:
        struct = dict_to_structure(data_dict)
        e_hull = _safe_alignn_predict("jv_ehull_alignn", pmg_to_jarvis(struct))

        e_hull = max(0.0, e_hull)
        data_dict["ehull_eV"] = e_hull

        # if e_hull <= 0.05:
        #     data_dict["synthesizability"] = "High"
        # elif e_hull <= 0.15:
        #     data_dict["synthesizability"] = "Medium"
        # else:
        #     data_dict["synthesizability"] = "Low"
    except Exception as e:
        print(f"[E_hull Error]: {e}")
        data_dict["ehull_eV"] = 999.0
        # data_dict["synthesizability"] = "Error"
    return data_dict


def characterize_composition_and_symmetry(data_dict: dict) -> dict:
    """
    提取化学式、空间群符号及编号、以及元素组成。
    """
    try:
        struct = dict_to_structure(data_dict)
        comp = struct.composition

        # 1. 基础信息
        data_dict["formula"] = comp.reduced_formula
        data_dict["num_elements"] = len(comp.elements)
        data_dict["element_set"] = [str(e) for e in comp.elements]

        # 2. 提取空间群
        # symprec=0.1 允许一定的坐标弛豫误差
        sg_symbol, sg_number = struct.get_space_group_info(symprec=0.1)
        data_dict["spacegroup"] = f"{sg_symbol} ({sg_number})"

        # 【新增】：单独提取空间群编号，作为整型，方便做约束！
        data_dict["sg_number"] = sg_number

    except Exception as e:
        data_dict["formula"] = "Unknown"
        data_dict["spacegroup"] = "Unknown"
        data_dict["sg_number"] = -1  # 异常赋予 -1
        data_dict["num_elements"] = 999
        data_dict["element_set"] = []

    return data_dict

def characterize_jacs_clscore(data_dict: dict) -> dict:
    try:
        # 每次调用时连一次代理，非常轻量安全
        jacs_proxy = xmlrpc.client.ServerProxy("http://localhost:8080/")

        struct = dict_to_structure(data_dict)
        cif_string = struct.to(fmt="cif")
        cl_score = jacs_proxy.predict_cif(cif_string)
        data_dict["cl_score"] = cl_score
    except Exception as e:
        print(f"\n❌[JACS CLscore 预测失败]: {e}")
        data_dict["cl_score"] = -1.0
    return data_dict


def characterize_piezoelectric(data_dict: dict) -> dict:
    try:
        struct = dict_to_structure(data_dict)
        sga = SpacegroupAnalyzer(struct)
        symm_ops = sga.get_symmetry_operations()
        is_centrosymmetric = False
        for op in symm_ops:
            if np.allclose(op.rotation_matrix, -np.eye(3)):
                is_centrosymmetric = True
                break
        data_dict["has_piezo_potential"] = not is_centrosymmetric
    except Exception:
        data_dict["has_piezo_potential"] = False
    return data_dict

def characterize_anisotropy(data_dict: dict, ratio_threshold: float = 1.05) -> dict:
    try:
        struct = dict_to_structure(data_dict)
        _, _, _ = reconstruct_crystal_geometry(struct) # 借用这步测真空方向，其实有更快的写法，但这样稳妥
        # 简单判定：直接找晶格的 a, b, c
        abc = struct.lattice.abc
        max_idx = np.argmax(abc)
        in_plane = [abc[i] for i in range(3) if i != max_idx]
        ratio = max(in_plane) / min(in_plane)
        data_dict["in_plane_anisotropy_ratio"] = ratio
        data_dict["is_anisotropic"] = bool(ratio > ratio_threshold)
    except Exception:
        data_dict["in_plane_anisotropy_ratio"] = 1.0
        data_dict["is_anisotropic"] = False
    return data_dict


from mattergen.evaluation.utils.relaxation import relax_structures
import torch
import numpy as np


def relax_crystal_geometry_mattersim(data_dict: dict, max_steps: int = 50) -> dict:
    """
    使用微软官方的 MatterSim 进行结构弛豫。
    将 MatterGen 生成的粗糙结构，精修为符合 MatterSim 势函数的完美极小态。
    """
    try:
        # 1. 提取未经优化的毛坯房结构
        struct = dict_to_structure(data_dict)

        # 2. 调用 MatterSim 官方弛豫接口
        # 传递 steps=max_steps 给底层的 ASE 优化器 (LBFGS/FIRE)
        # 如果你的 kwargs 名字不同，请根据实际使用的 ase 参数调整，通常 fmax=0.05 也是支持的
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.enable_grad():
            with suppress_output():
                relaxed_structs, total_energies = relax_structures(
                    structures=[struct],
                    device=device_str,
                    # max_n_steps=max_steps
                )

        relaxed_struct = relaxed_structs[0]
        final_total_energy = total_energies[0]

        # 3. 覆盖原始结构缓存
        data_dict["structure_obj"] = relaxed_struct

        # 同步更新底层的张量坐标 (为了保存 CIF 时也是优化后的结构)
        data_dict["pos"] = torch.tensor(relaxed_struct.frac_coords, dtype=torch.float32)
        data_dict["cell"] = torch.tensor(relaxed_struct.lattice.matrix, dtype=torch.float32).unsqueeze(0)

        # 记录 MatterSim 算出的每原子能量 (作为参考，因为它是总能量，需要除以原子数)
        data_dict["mattersim_energy_eV"] = float(final_total_energy) / len(relaxed_struct)

        # 4. 【顺手测受力】：因为 MatterSim 接口没直接返回 fmax，我们用常驻内存的 CHGNet 测一下
        # 这一步是为了给约束条件 "f_max_eV_A" 提供数据，过滤掉那些即使弛豫了也烂泥扶不上墙的垃圾
        with torch.enable_grad():
            pred = chgnet_model.predict_structure(relaxed_struct)
            forces = np.array(pred.get('f', []))
            stress = np.array(pred.get('s', []))

            f_max = float(np.max(np.linalg.norm(forces, axis=1))) if forces.size > 0 else 999.0
            # VASP 单位转 GPa 约乘 160.21
            s_max = float(np.max(np.abs(stress))) * 160.21766 if stress.size > 0 else 999.0

            data_dict["f_max_eV_A"] = f_max
            data_dict["s_max_GPa"] = s_max

    except Exception as e:
        print(f"MatterSim 结构优化崩溃: {e}")
        pass

    return data_dict

def extract_all_properties(data_dict: dict) -> dict:
    _ = dict_to_structure(data_dict)

    # 1. 【原汤化原食】：全自动 MatterSim 结构弛豫！
    if "mattersim_energy_eV" not in data_dict:
        data_dict = relax_crystal_geometry_mattersim(data_dict, max_steps=50)

    # --- 0. 基础信息 (化学式与空间群) ---
    if "formula" not in data_dict:
        data_dict = characterize_composition_and_symmetry(data_dict)

    if "formation_energy_eV" not in data_dict:
        data_dict = characterize_formation_energy_alignn(data_dict)
    if "exfoliation_energy_meV" not in data_dict:
        data_dict = characterize_exfoliation_and_vacuum(data_dict)
    if "f_max_eV_A" not in data_dict:
        data_dict = characterize_chgnet_basic(data_dict)
    if "bandgap_eV" not in data_dict:
        data_dict = characterize_bandgap(data_dict)
    if "dielectric_epsx" not in data_dict:
        data_dict = characterize_dielectric_constant(data_dict)
    if "ehull_eV" not in data_dict:
        data_dict = characterize_synthesizability_ehull(data_dict)

    # --- 加上下面这三个！ ---
    if "cl_score" not in data_dict:
        data_dict = characterize_jacs_clscore(data_dict)
    if "has_piezo_potential" not in data_dict:
        data_dict = characterize_piezoelectric(data_dict)
    if "is_anisotropic" not in data_dict:
        data_dict = characterize_anisotropy(data_dict)

    return data_dict


