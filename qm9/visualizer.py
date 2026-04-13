import torch
import numpy as np
import os
import glob
import random
import matplotlib
import imageio

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qm9 import bond_analyze
##############
### Files ####
###########-->


def save_xyz_file(path, one_hot, charges, positions, dataset_info, id_from=0, name='molecule', node_mask=None):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        f.write("%d\n\n" % atomsxmol[batch_i])
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = dataset_info['atom_decoder'][atom]
            f.write("%s %.9f %.9f %.9f\n" % (atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
        f.close()


def load_molecule_xyz(file, dataset_info):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(dataset_info['atom_decoder']))
        charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, dataset_info['atom_encoder'][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot, charges


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files

#<----########
### Files ####
##############
def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    # for i in range(2):
    #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0,
                    alpha=alpha)
    # # calculate vectors for "vertical" circle
    # a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    # b = np.array([0, 1, 0])
    # b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (
    #             1 - np.cos(rot))
    # ax.plot(np.sin(u), np.cos(u), 0, color='k', linestyle='dashed')
    # horiz_front = np.linspace(0, np.pi, 100)
    # ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color='k')
    # vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    # ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u),
    #         a[2] * np.sin(u) + b[2] * np.cos(u), color='k', linestyle='dashed')
    # ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front),
    #         b[1] * np.cos(vert_front),
    #         a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front), color='k')
    #
    # ax.view_init(elev=elev, azim=0)


def _get_perp_unit_vector(v):
    """给定向量 v (3,), 返回与 v 垂直的单位向量（任一）。"""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    u = v / norm
    # 选一个“任意向量”，避免与 u 共线
    arb = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, arb)) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    perp = np.cross(u, arb)
    perp_norm = np.linalg.norm(perp)
    if perp_norm == 0:
        # 退化情况
        return np.array([1.0, 0.0, 0.0])
    return perp / perp_norm


import numpy as np


def _get_perp_unit_vector(v):
    """
    计算向量 v 的任意一个单位垂直向量。
    为了防止视觉上平行线在某些角度重叠严重，
    通常选一个相对固定的参考轴做叉积。
    """
    v = v / (np.linalg.norm(v) + 1e-6)

    # 尝试与 Z 轴做叉积
    cross_z = np.cross(v, np.array([0, 0, 1]))
    norm_z = np.linalg.norm(cross_z)

    # 如果 v 本身平行于 Z 轴（叉积接近0），则改与 Y 轴做叉积
    if norm_z < 1e-3:
        perp = np.cross(v, np.array([0, 1, 0]))
    else:
        perp = cross_z

    return perp / (np.linalg.norm(perp) + 1e-6)


def draw_bond_lines(ax, p1, p2, bond_order=1, color='k', alpha=1.0, base_linewidth=2.0):
    """
    优化后的 3D 键绘制函数：
    1. 使用固定间距，避免线条粘连。
    2. 添加内缩 (shrink)，避免线条插入原子球体内部显得杂乱。
    """
    if bond_order <= 0:
        return

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    length = np.linalg.norm(v)

    if length < 1e-3:
        return

    # --- 改进点 1: 内缩处理 (Shrink) ---
    # 如果画了原子球，线最好不要直接连到球心，而是留出一点空间
    # 假设原子半径大约是 0.3~0.4 (根据你的数据调整)
    shrink_val = 0.0  # 如果没有画球体，设为0；如果画了球体，设为 0.2 或 0.3
    if length > 2 * shrink_val:
        dir_vec = v / length
        p1 = p1 + dir_vec * shrink_val
        p2 = p2 - dir_vec * shrink_val
        v = p2 - p1  # 更新向量

    # --- 改进点 2: 固定间距 (Fixed Offset) ---
    # 不要用 length * frac，而是用固定的物理距离
    # 0.15 ~ 0.25 通常在分子可视化中效果较好
    fixed_offset = 0.18

    perp_unit = _get_perp_unit_vector(v)
    offset_vec = perp_unit * fixed_offset

    n = int(bond_order)

    # 计算线条的偏移列表
    if n == 1:
        shifts = [0.0]
        linewidths = [base_linewidth]
    elif n == 2:
        # 双键：两边对称分开
        shifts = [-0.5, 0.5]
        linewidths = [base_linewidth * 0.9] * 2
    elif n == 3:
        # 三键：中间一条，两边两条
        shifts = [-1.0, 0.0, 1.0]
        # 中间稍微粗一点，两边细一点，增强层次感
        linewidths = [base_linewidth * 0.8, base_linewidth, base_linewidth * 0.8]
    elif n == 4:  # 也就是你代码里的 "1.5" (芳香键/共轭键)
        # 此时通常画一条实线，一条虚线，或者两条不等宽的线
        shifts = [-0.4, 0.4]
        linewidths = [base_linewidth, base_linewidth * 0.6]  # 一粗一细
    else:
        shifts = [0.0]
        linewidths = [base_linewidth]

    # 绘制
    for idx, w in zip(shifts, linewidths):
        shift_vec = offset_vec * idx
        p1s = p1 + shift_vec
        p2s = p2 + shift_vec

        ax.plot([p1s[0], p2s[0]],
                [p1s[1], p2s[1]],
                [p1s[2], p2s[2]],
                linewidth=w,
                c=color,
                alpha=alpha)


import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, bond_color,
                  dataset_info):
    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.7 * alpha,
                   c=colors)

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], \
                dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]],
                    dataset_info['atom_decoder'][s[1]])
            if 'qm9' in dataset_info['name']:
                draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
                line_width = (3 - 2) * 2 * 2
            elif dataset_info['name'] == 'geom':
                draw_edge_int = bond_analyze.geom_predictor(pair, dist)
                line_width = 2
            else:
                raise Exception('Wrong dataset_info name')

            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    linewidth_factor = 1

                p1 = np.array([x[i], y[i], z[i]])
                p2 = np.array([x[j], y[j], z[j]])
                # 此处使用了修改后的 bond_color
                draw_bond_lines(ax, p1, p2, bond_order=int(draw_edge_int),
                                color=bond_color, alpha=alpha,
                                base_linewidth=line_width * linewidth_factor)


def plot_data3d(positions, atom_type, dataset_info, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1.):
    # 统一将分子键的颜色设为 #8E8E8E
    bond_color = '#5E5E5E'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()

    # 1. 设置整个画布背景为透明
    fig.patch.set_alpha(0.0)

    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)

    # 2. 设置3D坐标系的背景为透明
    ax.set_facecolor('none')

    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    # 隐藏坐标轴线
    ax.xaxis.line.set_color("none")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                  bond_color, dataset_info)

    if 'qm9' in dataset_info['name']:
        max_value = positions.abs().max().item()
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = positions.abs().max().item()
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 200

    if save_path is not None:
        # 3. 增加 transparent=True，确保保存下来的 png 图片是带透明背景的
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)

        if spheres_3d:
            img = imageio.imread(save_path)
            # 4. 判断如果是含有 Alpha (透明度) 通道的 RGBA 图像，需要将 RGB 和 Alpha 分离出来提亮，以免把透明区域变黑
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_rgb = img[..., :3]
                img_alpha = img[..., 3:]
                img_brighter_rgb = np.clip(img_rgb * 1.4, 0, 255).astype('uint8')
                img_brighter = np.concatenate([img_brighter_rgb, img_alpha], axis=2)
            else:
                img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_data3d_highlight(positions, atom_type, dataset_info, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1., highlight_idx=None, view_elev=None, view_azim=None):
    """
    新增参数:
    highlight_idx: list or set, 需要高亮的原子索引列表 (e.g., [0, 1, 2])
    """
    black = (0, 0, 0)
    white = (1, 1, 1)
    # hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'
    hex_bg_color = '#5E5E5E'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)

    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.xaxis.line.set_color("black")
    else:
        ax.xaxis.line.set_color("white")

    # 将 highlight_idx 传递给 plot_molecule
    plot_molecule_highlight(ax, positions, atom_type, alpha, spheres_3d,
                  hex_bg_color, dataset_info, highlight_idx)

    # ... (设置坐标轴范围的代码保持不变) ...
    if 'qm9' in dataset_info['name']:
        max_value = positions.abs().max().item()
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = positions.abs().max().item()
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 300 if spheres_3d else 200



    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)

        if spheres_3d:
            img = imageio.imread(save_path)
            # 稍微调亮一点，避免高亮部分过于刺眼或者太暗
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
import imageio


def plot_data3d_highlight_triview(positions, atom_type, dataset_info, camera_elev=0, camera_azim=0, save_path=None,
                          spheres_3d=False,
                          bg='black', alpha=1., highlight_idx=None):
    """
    包装后的三视图绘制函数。
    形参完全保持不变，但在内部忽略 camera_elev/azim，强制生成 XY, XZ, YZ 三视图。
    """
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    # 提前计算坐标轴范围，保证三个视图比例一致
    if 'qm9' in dataset_info['name']:
        max_value = positions.abs().max().item()
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    elif dataset_info['name'] == 'geom':
        max_value = positions.abs().max().item()
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    else:
        raise ValueError(dataset_info['name'])

    from mpl_toolkits.mplot3d import Axes3D

    # --- 修改点 1: 创建宽画布以容纳三个子图 ---
    # figsize=(15, 5) 这里的比例 3:1 对应 3 个子图
    fig = plt.figure(figsize=(15, 5))

    # 定义三个视图的角度配置 (Elev, Azim, Title)
    # XY (俯视), XZ (正视), YZ (侧视)
    # 注意：Matplotlib 的 view_init 角度定义比较特殊，以下配置对应标准三视图
    views = [
        (90, -90, 'XY (Top)'),
        (0, -90, 'XZ (Front)'),
        (0, 0, 'YZ (Side)')
    ]

    # --- 修改点 2: 循环创建子图并绘制 ---
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # 强制使用正交投影 (无透视，适合三视图)，如果喜欢透视感可注释掉此行
        ax.set_proj_type('ortho')

        ax.set_aspect('auto')
        ax.view_init(elev=elev, azim=azim)

        # 设置背景和隐藏坐标轴
        if bg == 'black':
            ax.set_facecolor(black)
            ax.xaxis.line.set_color("black")  # 其实下面隐藏了，这行主要防漏
        else:
            ax.set_facecolor(white)
            ax.xaxis.line.set_color("white")

        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        ax._axis3don = False

        # 调用核心绘图逻辑 (复用之前的 plot_molecule_highlight)
        plot_molecule_highlight(ax, positions, atom_type, alpha, spheres_3d,
                                hex_bg_color, dataset_info, highlight_idx)

        # 设置统一的坐标轴范围
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)

        # 可选：如果你想给每个视图加标题，可以取消注释下面这行
        # ax.set_title(title, color='white' if bg=='black' else 'black')

    dpi = 120 if spheres_3d else 200

    if save_path is not None:
        # 保存整张画布 (包含三个子图)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=dpi)  # pad_inches稍微给一点，避免子图太挤

        if spheres_3d:
            try:
                img = imageio.imread(save_path)
                # 稍微调亮一点
                img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
                imageio.imsave(save_path, img_brighter)
            except Exception as e:
                print(f"Warning: Could not post-process image with imageio: {e}")
    else:
        plt.show()

    plt.close()


def plot_molecule_highlight(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color,
                            dataset_info, highlight_idx=None):
    # --- 0. 设置 3D 背景为完全透明 ---
    # 隐藏 3D 坐标系的背景面板 (x, y, z 面)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # 设置 axes 背景色为透明
    ax.set_facecolor('none')
    # (可选) 如果你连坐标轴的线和刻度都不想要，可以直接关掉：
    # ax.axis('off')

    # --- 1. 准备高亮配置 ---
    highlight_set = set(highlight_idx) if highlight_idx is not None else set()

    highlight_color_hex = '#5232c7'
    highlight_color_rgb = (1.0, 0.84, 0.0)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    # --- 2. 绘制原子 ---
    if spheres_3d:
        for idx, (i, j, k, s, c) in enumerate(zip(x, y, z, radii, colors)):
            # 2.1 绘制底层的普通的 3D 实心球
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)

            # 2.2 绘制高亮球体的 3D 纹理 (网格)
            if idx in highlight_set:
                r_high = 0.7 * s * 1.15

                # --- 修改这里：降低生成球面的分辨率，使网格变疏 ---
                # 原来是 16 和 12，现在改成 10 和 7 (你可以根据视觉效果继续微调这两个数值)
                u = np.linspace(0, 2 * np.pi, 10)  # 经度 (竖线数量)
                v = np.linspace(0, np.pi, 7)  # 纬度 (横线数量)

                X = r_high * np.outer(np.cos(u), np.sin(v)) + i.item()
                Y = r_high * np.outer(np.sin(u), np.sin(v)) + j.item()
                Z = r_high * np.outer(np.ones(np.size(u)), np.cos(v)) + k.item()

                # 绘制稀疏的 3D 网格线框
                ax.plot_wireframe(X, Y, Z,
                                  color=highlight_color_hex,
                                  alpha=0.7,  # 可以略微调高透明度让线条更实
                                  linewidth=1.5,  # 稍微加粗一点线条，配合稀疏网格更好看
                                  zorder=3)

                # --- B. (可选) 如果你希望除了纹理，还要保留原先那层黄色的半透明高亮光晕，可以把下面解开 ---
                # ax.plot_surface(X, Y, Z,
                #                 color=highlight_color_rgb,
                #                 alpha=0.2,    # 透明底色
                #                 shade=False,
                #                 zorder=2)

    else:
        # 2.1 绘制底层普通原子 (实心)
        # 无论是否高亮，先画出原本的原子，这样高亮的纹理会叠加在颜色之上
        n_atoms = len(x)
        # 创建布尔数组：哪些是高亮，哪些是普通
        is_highlight = np.array([i in highlight_set for i in range(n_atoms)])
        is_normal = ~is_highlight  # 取反

        # 2.2 绘制普通原子 (Normal Atoms)
        if np.any(is_normal):
            ax.scatter(x[is_normal], y[is_normal], z[is_normal],
                       s=areas[is_normal],
                       c=colors[is_normal],
                       alpha=0.7 * alpha,
                       edgecolors='none',
                       zorder=1)  # 层级最低

        # 2.3 绘制高亮原子 (Highlighted Atoms)
        if np.any(is_highlight):
            # A. 先画底色 (Base Color) - 确保高亮原子本身可见
            ax.scatter(x[is_highlight], y[is_highlight], z[is_highlight],
                       s=areas[is_highlight],
                       c=colors[is_highlight],
                       alpha=0.7 * alpha,
                       edgecolors='none',
                       zorder=2)  # 层级比普通原子高

            # B. 再画纹理 (Texture / Hatch) - 叠加在底色之上
            # 技巧：zorder设得比底色稍微高一点点，或者利用绘制顺序
            ax.scatter(x[is_highlight], y[is_highlight], z[is_highlight],
                       s=areas[is_highlight],
                       facecolors='none',  # 透明背景
                       edgecolors=highlight_color_hex,
                       hatch='//',  # 纹理
                       linewidths=1,
                       alpha=1.0,  # 纹理不透明度高一点
                       zorder=3)  # 层级最高，强制在最上层

    # --- 3. 绘制化学键 ---
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))

            atom1 = dataset_info['atom_decoder'][atom_type[i]]
            atom2 = dataset_info['atom_decoder'][atom_type[j]]

            # ... (获取 bond_order 的逻辑保持不变) ...
            if 'qm9' in dataset_info['name']:
                draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
                line_width = (3 - 2) * 2 * 2
            elif dataset_info['name'] == 'geom':
                draw_edge_int = bond_analyze.geom_predictor((atom1, atom2), dist)  # 注意这里 pair 参数可能需要调整
                line_width = 2
            else:
                draw_edge_int = 1  # Fallback
                line_width = 2

            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    linewidth_factor = 1

                # 【高亮逻辑 - 样式调整】
                if i in highlight_set and j in highlight_set:
                    bond_color = highlight_color_hex
                    current_alpha = 1.0
                    current_lw = line_width * linewidth_factor * 1.5
                    # 设置高亮键为虚线，模拟"纹理感"
                    bond_style = '-'
                else:
                    bond_color = hex_bg_color
                    current_alpha = alpha
                    current_lw = line_width * linewidth_factor
                    bond_style = '-'  # 普通键为实线

                draw_bond_lines_highlight(ax, p1, p2, bond_order=int(draw_edge_int),
                                color=bond_color, alpha=current_alpha,
                                base_linewidth=current_lw,
                                linestyle=bond_style)  # 传入 linestyle


# --- 需要同步更新 draw_bond_lines 函数以支持 linestyle ---
def draw_bond_lines_highlight(ax, p1, p2, bond_order=1, color='k', alpha=1.0,
                    base_linewidth=2.0, linestyle='-'):
    """
    增加了 linestyle 参数
    """
    if bond_order <= 0: return

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    length = np.linalg.norm(v)

    if length < 1e-3: return

    # 内缩处理
    shrink_val = 0.0
    if length > 2 * shrink_val:
        dir_vec = v / length
        p1 = p1 + dir_vec * shrink_val
        p2 = p2 - dir_vec * shrink_val
        v = p2 - p1

    fixed_offset = 0.18
    perp_unit = _get_perp_unit_vector(v) # 确保这个辅助函数存在
    offset_vec = perp_unit * fixed_offset

    n = int(bond_order)

    # 偏移逻辑
    if n == 1:
        shifts = [0.0]; linewidths = [base_linewidth]
    elif n == 2:
        shifts = [-0.5, 0.5]; linewidths = [base_linewidth * 0.9] * 2
    elif n == 3:
        shifts = [-1.0, 0.0, 1.0]; linewidths = [base_linewidth * 0.8, base_linewidth, base_linewidth * 0.8]
    elif n == 4:
        shifts = [-0.4, 0.4]; linewidths = [base_linewidth, base_linewidth * 0.6]
    else:
        shifts = [0.0]; linewidths = [base_linewidth]

    for idx, w in zip(shifts, linewidths):
        shift_vec = offset_vec * idx
        p1s = p1 + shift_vec
        p2s = p2 + shift_vec

        ax.plot([p1s[0], p2s[0]],
                [p1s[1], p2s[1]],
                [p1s[2], p2s[2]],
                linewidth=w,
                c=color,
                alpha=alpha,
                linestyle=linestyle) # 【应用虚线样式】


def plot_data3d_uncertainty(
        all_positions, all_atom_types, dataset_info, camera_elev=0, camera_azim=0,
        save_path=None, spheres_3d=False, bg='black', alpha=1.):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.xaxis.line.set_color("black")
    else:
        ax.xaxis.line.set_color("white")

    for i in range(len(all_positions)):
        positions = all_positions[i]
        atom_type = all_atom_types[i]
        plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                      hex_bg_color, dataset_info)

    if 'qm9' in dataset_info['name']:
        max_value = all_positions[0].abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    elif dataset_info['name'] == 'geom':
        max_value = all_positions[0].abs().max().item()

        # axis_lim = 3.2
        axis_lim = min(40, max(max_value / 2 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(dataset_info['name'])

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_grid():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.

        ax.imshow(im)

    plt.show()


def visualize(path, dataset_info, max_num=25, wandb=None, spheres_3d=False):
    files = load_xyz_files(path)[0:max_num]
    for file in files:
        positions, one_hot, charges = load_molecule_xyz(file, dataset_info)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
        dists = dists[dists > 0]
        print("Average distance between atoms", dists.mean().item())
        plot_data3d(positions, atom_type, dataset_info=dataset_info, save_path=file[:-4] + '.png',
                    spheres_3d=spheres_3d)

        if wandb is not None:
            path = file[:-4] + '.png'
            # Log image(s)
            im = plt.imread(path)
            wandb.log({'molecule': [wandb.Image(im, caption=path)]})


def visualize_chain(path, dataset_info, wandb=None, spheres_3d=False,
                    mode="chain"):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    for i in range(len(files)):
        file = files[i]

        positions, one_hot, charges = load_molecule_xyz(file, dataset_info=dataset_info)

        atom_type = torch.argmax(one_hot, dim=1).numpy()
        fn = file[:-4] + '.png'
        plot_data3d(positions, atom_type, dataset_info=dataset_info,
                    save_path=fn, spheres_3d=spheres_3d, alpha=1.0)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


def visualize_chain_uncertainty(
        path, dataset_info, wandb=None, spheres_3d=False, mode="chain"):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    for i in range(len(files)):
        if i + 2 == len(files):
            break

        file = files[i]
        file2 = files[i+1]
        file3 = files[i+2]

        positions, one_hot, _ = load_molecule_xyz(file, dataset_info=dataset_info)
        positions2, one_hot2, _ = load_molecule_xyz(
            file2, dataset_info=dataset_info)
        positions3, one_hot3, _ = load_molecule_xyz(
            file3, dataset_info=dataset_info)

        all_positions = torch.stack([positions, positions2, positions3], dim=0)
        one_hot = torch.stack([one_hot, one_hot2, one_hot3], dim=0)

        all_atom_type = torch.argmax(one_hot, dim=2).numpy()
        fn = file[:-4] + '.png'
        plot_data3d_uncertainty(
            all_positions, all_atom_type, dataset_info=dataset_info,
            save_path=fn, spheres_3d=spheres_3d, alpha=0.5)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


if __name__ == '__main__':
    #plot_grid()
    import qm9.dataset as dataset
    from configs.datasets_config import qm9_with_h, geom_with_h
    matplotlib.use('macosx')

    task = "visualize_molecules"
    task_dataset = 'geom'

    if task_dataset == 'qm9':
        dataset_info = qm9_with_h

        class Args:
            batch_size = 1
            num_workers = 0
            filter_n_atoms = None
            datadir = 'qm9/temp'
            dataset = 'qm9'
            remove_h = False
            include_charges = True

        cfg = Args()

        dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

        for i, data in enumerate(dataloaders['train']):

            positions = data['positions'].view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = data['one_hot'].view(-1, 5).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            plot_data3d(
                positions_centered, atom_type, dataset_info=dataset_info,
                spheres_3d=True)

    elif task_dataset == 'geom':
        files = load_xyz_files('outputs/data')
        matplotlib.use('macosx')
        for file in files:
            x, one_hot, _ = load_molecule_xyz(file, dataset_info=geom_with_h)

            positions = x.view(-1, 3)
            positions_centered = positions - positions.mean(dim=0, keepdim=True)
            one_hot = one_hot.view(-1, 16).type(torch.float32)
            atom_type = torch.argmax(one_hot, dim=1).numpy()

            mask = (x == 0).sum(1) != 3
            positions_centered = positions_centered[mask]
            atom_type = atom_type[mask]

            plot_data3d(
                positions_centered, atom_type, dataset_info=geom_with_h,
                spheres_3d=False)

    else:
        raise ValueError(dataset)
