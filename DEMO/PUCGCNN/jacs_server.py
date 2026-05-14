import os
import shutil
import tempfile
import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from xmlrpc.server import SimpleXMLRPCServer

# 引入官方库模块
from cgcnn.data_PU_learning import CIFData, collate_pool
from cgcnn.model_PU_learning import CrystalGraphConvNet

# ================= 全局内存缓存 =================
MODELS_CACHE = []
MODEL_DIR = ""
ATOM_INIT_PATH = ""


def predict_cif(cif_string: str) -> float:
    """
    接收来自主程序的 CIF 字符串，在内存中动态建图并使用 100 个常驻模型推理。
    """
    global MODELS_CACHE

    # 在 Linux 内存盘创建微小的沙盒
    base_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    temp_dir = tempfile.mkdtemp(prefix="jacs_rpc_", dir=base_dir)

    try:
        # 直接把传进来的 CIF 字符串写入临时沙盒
        with open(os.path.join(temp_dir, "target.cif"), "w") as f:
            f.write(cif_string)
        with open(os.path.join(temp_dir, "id_prop.csv"), "w") as f:
            f.write("target,0\n")
        shutil.copy(ATOM_INIT_PATH, temp_dir)

        # 动态建图
        dataset = CIFData(temp_dir)
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_pool)
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

        # 【核心逻辑】：懒加载。只在第一次收到请求时，把 100 个模型加载到内存死死锁住！
        if not MODELS_CACHE:
            print("\n[Server] 首次收到请求，正在将 100 个集成模型永久载入内存...")
            for i in range(1, 101):
                modelpath = os.path.join(MODEL_DIR, f'checkpoint_bag_{i}.pth.tar')
                if not os.path.isfile(modelpath): continue

                checkpoint = torch.load(modelpath, map_location='cpu')
                model_args = argparse.Namespace(**checkpoint['args'])

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
                MODELS_CACHE.append(model)
            print(f"[Server] {len(MODELS_CACHE)} 个模型加载完毕！极速响应模式已开启。")

        # 内存轮询：瞬间跑完 100 个模型
        ensemble_probs = []
        for model in MODELS_CACHE:
            with torch.no_grad():
                output = model(*input_var)
                prob = torch.exp(output)[0, 1].item()
                ensemble_probs.append(prob)

        return float(np.mean(ensemble_probs))

    except Exception as e:
        print(f"[Server Error] {e}")
        return -1.0
    finally:
        # 清理沙盒
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="./trained_models")
    parser.add_argument('--atom_init', type=str, default="./atom_init.json")
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    ATOM_INIT_PATH = args.atom_init

    # 启动轻量级 RPC 服务器
    server = SimpleXMLRPCServer(("localhost", args.port), allow_none=True)
    server.register_function(predict_cif, "predict_cif")
    print(f"✅ JACS CLscore AI 引擎已启动，监听端口 {args.port}...")
    print("等待主程序发送数据...")
    server.serve_forever()