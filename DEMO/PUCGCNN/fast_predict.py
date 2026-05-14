import os
import sys
import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

# 引入官方库的模块
from cgcnn.data_PU_learning import CIFData, collate_pool
from cgcnn.model_PU_learning import CrystalGraphConvNet


def predict_single_cif(cif_dir, model_dir, bag_size=100):
    """
    极速推理函数：在内存中动态建图，连续通过 100 个模型，直接输出均值。
    """
    # 1. 利用官方的 CIFData 直接从文件夹动态建图 (不需要生成 pickle)
    dataset = CIFData(cif_dir)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_pool)

    # 获取唯一的那个晶体图数据
    inputs_tuple, target, cif_id = next(iter(loader))

    with torch.no_grad():
        input_var = (
            Variable(inputs_tuple[0]),
            Variable(inputs_tuple[1]),
            inputs_tuple[2],
            inputs_tuple[3]
        )

    orig_atom_fea_len = inputs_tuple[0].shape[-1]
    nbr_fea_len = inputs_tuple[1].shape[-1]

    ensemble_probs = []

    # 2. 内存轮询：加载 100 个模型进行推理
    for i in range(1, bag_size + 1):
        modelpath = os.path.join(model_dir, f'checkpoint_bag_{i}.pth.tar')
        if not os.path.isfile(modelpath):
            continue

        # 强制使用 CPU 加载（极大地提升单次推理的启动速度）
        checkpoint = torch.load(modelpath, map_location='cpu')
        model_args = argparse.Namespace(**checkpoint['args'])

        # 实例化模型
        model = CrystalGraphConvNet(
            orig_atom_fea_len, nbr_fea_len,
            atom_fea_len=model_args.atom_fea_len,
            n_conv=model_args.n_conv,
            h_fea_len=model_args.h_fea_len,
            n_h=model_args.n_h,
            classification=True
        )

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # 核心推理代码：计算概率
        with torch.no_grad():
            output = model(*input_var)
            # 输出是 log_softmax，用 exp 还原，[0, 1] 里的 1 代表可合成
            prob = torch.exp(output)[0, 1].item()
            ensemble_probs.append(prob)

    if not ensemble_probs:
        return -1.0

    # 3. 返回 100 个模型的平均预测值 (Bagging 思想)
    return np.mean(ensemble_probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_dir', type=str, required=True, help="包含 target.cif 和 id_prop.csv 的目录")
    parser.add_argument('--model_dir', type=str, required=True, help="100个预训练模型的存放目录")
    args = parser.parse_args()

    # 屏蔽一切杂乱输出，只打印最终分数
    import warnings

    warnings.filterwarnings("ignore")

    final_score = predict_single_cif(args.cif_dir, args.model_dir)
    print(f"CLSCORE:{final_score:.6f}")