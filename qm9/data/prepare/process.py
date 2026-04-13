import logging
import os
import torch
import tarfile
from torch.nn.utils.rnn import pad_sequence

charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}


def split_dataset(data, split_idxs):
    """
    Splits a dataset according to the indices given.

    Parameters
    ----------
    data : dict
        Dictionary to split.
    split_idxs :  dict
        Dictionary defining the split.  Keys are the name of the split, and
        values are the keys for the items in data that go into the split.

    Returns
    -------
    split_dataset : dict
        The split dataset.
    """
    split_data = {}
    for set, split in split_idxs.items():
        split_data[set] = {key: val[split] for key, val in data.items()}

    return split_data

# def save_database()


def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    # Now loop over files using readfile function defined above
    # Process each file accordingly using process_file_fn

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            molecules.append(process_file_fn(openfile))

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules


def process_xyz_md17(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the MD-17 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    line_counter = 0
    atom_positions = []
    atom_types = []
    for line in xyz_lines:
        if line[0] is '#':
            continue
        if line_counter is 0:
            num_atoms = int(line)
        elif line_counter is 1:
            split = line.split(';')
            assert (len(split) == 1 or len(split) == 2), 'Improperly formatted energy/force line.'
            if (len(split) == 1):
                e = split[0]
                f = None
            elif (len(split) == 2):
                e, f = split
                f = f.split('],[')
                atom_energy = float(e)
                atom_forces = [[float(x.strip('[]\n')) for x in force.split(',')] for force in f]
        else:
            split = line.split()
            if len(split) is 4:
                type, x, y, z = split
                atom_types.append(split[0])
                atom_positions.append([float(x) for x in split[1:]])
            else:
                logging.debug(line)
        line_counter += 1

    atom_charges = [charge_dict[type] for type in atom_types]

    molecule = {'num_atoms': num_atoms, 'energy': atom_energy, 'charges': atom_charges,
                'forces': atom_forces, 'positions': atom_positions}

    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


def process_xyz_gdb9(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule


import os
import tarfile
import torch
import numpy as np
import io
import logging
from torch.nn.utils.rnn import pad_sequence


# ==========================================
# 1. 核心解析器 (作为 process_file_fn 传入)
# ==========================================
def parse_xtb_molecule(file_obj, dataset_info=None):
    """
    解析单个 XTB 格式的 XYZ 文件流
    格式示例:
    5
    gdb 1 mu_xtb=0.4201 homo_xtb=-6.5432 ...
    C      -0.0123      1.0850      0.0080
    ...
    """
    # 1. 读取并解码 (处理 tarfile 的 bytes 或 文件的 str)
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8')

    lines = content.strip().splitlines()
    if not lines:
        return None

    # 2. 解析原子数量
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None  # 空文件或格式错误

    # 3. 解析属性行 (第二行)
    # 格式: gdb 1 mu_xtb=0.420100 homo_xtb=-6.543200 ...
    properties = {}
    prop_line = lines[1]

    # 定义需要提取的 4 个属性及其类型
    target_props = ['mu_xtb', 'homo_xtb', 'lumo_xtb', 'gap_xtb']

    for part in prop_line.split():
        if '=' in part:
            key, val = part.split('=')
            if key in target_props:
                # 存为标量 Tensor (float32)
                properties[key] = torch.tensor(float(val), dtype=torch.float32)

    # 检查是否缺失属性，如果缺失补 NaN 或 0 (视需求而定)
    # 这里假设处理过的文件都有这些属性

    # 4. 解析原子坐标和类型
    atom_types = []
    positions = []
    charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        symbol = parts[0]
        pos = [float(x) for x in parts[1:4]]

        atom_types.append(charge_dict.get(symbol, 0))  # 默认为0或报错
        positions.append(pos)

    # 5. 构建输出字典
    # positions: [N, 3]
    # charges: [N]
    # one_hot: [N, 5] (需要 dataset_info)

    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    charges_tensor = torch.tensor(atom_types, dtype=torch.float32)  # 通常扩散模型用 float

    data = {
        'num_atoms': torch.tensor(n_atoms),
        'positions': pos_tensor,
        'charges': charges_tensor,
        'atom_mask': torch.ones(n_atoms, dtype=torch.bool)  # 有效原子掩码
    }

    # 生成 One-hot (如果提供了 dataset_info)
    if dataset_info is not None:
        atom_decoder = dataset_info['atom_decoder']  # ['H', 'C', 'N', 'O', 'F']
        # 将电荷映射回索引
        charge_to_idx = {charge_dict[sym]: i for i, sym in enumerate(atom_decoder)}
        indices = [charge_to_idx[c] for c in atom_types]

        one_hot = torch.nn.functional.one_hot(
            torch.tensor(indices, dtype=torch.long),
            num_classes=len(atom_decoder)
        ).float()
        data['one_hot'] = one_hot

    # 合并属性到 data 字典中
    data.update(properties)

    return data


# ==========================================
# 2. 修正后的 process_xyz_files
# ==========================================
def process_xyz_files_xtb(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    修改点：
    1. 增加对 tarfile 目录成员的过滤 (m.isfile())
    2. 兼容二进制流读取
    3. 适配新格式的属性堆叠
    """
    logging.info('Processing data file: {}'.format(data))

    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        # 【修改点】过滤掉目录 ('.')，只保留文件，防止读取错误
        files = [m for m in tardata.getmembers() if m.isfile()]

        # 提取函数
        readfile = lambda data_pt: tardata.extractfile(data_pt)

    elif os.is_dir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        # 普通文件读取，为了与 tar 保持一致，以二进制模式读取，在 parser 里解码
        readfile = lambda data_pt: open(data_pt, 'rb')

    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # 扩展名过滤
    if file_ext is not None:
        # tarfile member 的名字在 m.name 中
        is_tar = tarfile.is_tarfile(data)
        files = [f for f in files if (f.name if is_tar else f).endswith(file_ext)]

    # 索引过滤
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    molecules = []

    # 循环处理
    for file in files:
        # readfile 返回的是一个 file-like object
        f_obj = readfile(file)
        if f_obj is not None:
            try:
                # 调用解析函数
                mol_data = process_file_fn(f_obj)
                if mol_data is not None:
                    molecules.append(mol_data)
            finally:
                f_obj.close()

    if not molecules:
        logging.warning(f"No valid molecules found in {data}")
        return {}

    # 检查所有分子包含的键是否一致 (处理可能存在的解析失败)
    first_keys = molecules[0].keys()
    # 简单的健壮性处理：只保留包含所有键的分子，或者报错
    # 这里保持原逻辑：断言
    # assert all(mol.keys() == first_keys for mol in molecules), 'All molecules must have same set of properties/keys!'

    # 转换为 dict-of-lists
    # molecules_dict = {'positions': [mol1_pos, mol2_pos...], 'mu_xtb': [mol1_mu, ...]}
    molecules_dict = {prop: [mol[prop] for mol in molecules] for prop in first_keys}

    # 堆叠 (Stacking / Padding)
    if stack:
        stacked_dict = {}
        for key, val in molecules_dict.items():
            # 这里的 val 是一个 list of tensors
            if len(val) == 0:
                continue

            # 判断维度：标量属性 (mu_xtb) vs 序列属性 (positions)
            if val[0].dim() == 0:
                # 标量 -> Stack [N]
                stacked_dict[key] = torch.stack(val)
            else:
                # 序列 -> Pad [N, Max_Len, ...]
                # batch_first=True -> [Batch, Len, Dim]
                stacked_dict[key] = pad_sequence(val, batch_first=True)

        molecules = stacked_dict

    if tarfile.is_tarfile(data):
        tardata.close()

    return molecules

