import sys
import subprocess

# 尝试导入解析依赖所需的库，如果没有则自动安装
try:
    from packaging.requirements import Requirement
except ImportError:
    print("正在安装必要的环境检测库 (packaging)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    from packaging.requirements import Requirement

from importlib.metadata import version, PackageNotFoundError

# 你的包依赖列表
REQUIREMENTS =[
    "ase==3.26.0",
    "autopep8",
    "cachetools",
    "contextlib2",
    "emmet-core>=0.84.2",  # keep up-to-date together with pymatgen, atomate2
    "fire", # see https://github.com/google/python-fire
    "huggingface-hub",
    "hydra-core==1.3.1",
    "hydra-joblib-launcher==1.1.5",
    "jupyterlab>=4.2.5",
    "lmdb",
    "matplotlib==3.8.4",
    "matscipy>=0.7.0",
    "mattersim==1.2.3",
    "monty==2025.3.3",  # keep up-to-date together with pymatgen, atomate2
    "notebook>=7.2.2",
    "numpy<2.0",  # pin numpy before breaking changes in 2.0
    "omegaconf==2.3.0",
    "pymatgen>=2024.6.4",
    "pylint",
    "pytest",
    "pytorch-lightning==2.0.6",
    "seaborn>=0.13.2",  # for plotting
    "setuptools<81",
    "SMACT",
    "sympy>=1.11.1",
    "torch==2.2.1+cu118; sys_platform == 'linux'", 
    "torchvision==0.17.1+cu118; sys_platform == 'linux'",
    "torchaudio==2.2.1+cu118; sys_platform == 'linux'",
    "torch==2.4.1; sys_platform == 'darwin'",
    "torchvision==0.19.1; sys_platform == 'darwin'",
    "torchaudio==2.4.1; sys_platform == 'darwin'",
    "torch_cluster",
    "torch_geometric>=2.5",
    "torch_scatter",
    "torch_sparse",
    "tqdm",
    "wandb>=0.10.33",
]

def check_and_install():
    to_install =[]
    
    print("="*50)
    print("🔍 开始检测当前 Conda 环境依赖库...")
    print("="*50)
    
    for line in REQUIREMENTS:
        # 去除注释和多余的空格
        req_str = line.split("#")[0].strip()
        if not req_str:
            continue
            
        # 使用官方工具解析依赖字符串
        req = Requirement(req_str)
        
        # 1. 检查环境变量 (判断是 Mac 'darwin' 还是 Linux 'linux')
        if req.marker and not req.marker.evaluate():
            # 不符合当前系统的依赖直接跳过，比如在 Linux 下跳过 darwin 的包
            continue
            
        try:
            # 获取当前环境中已安装的版本
            installed_version = version(req.name)
            
            # 2. 对比版本号
            if req.specifier:
                if installed_version in req.specifier:
                    print(f"✅ {req.name:20s} | 已安装: {installed_version:12s} | 满足要求: {req.specifier}")
                else:
                    print(f"❌ {req.name:20s} | 已安装: {installed_version:12s} | 不满足: {req.specifier} (将重新安装)")
                    to_install.append(req_str)
            else:
                print(f"✅ {req.name:20s} | 已安装: {installed_version:12s} | 无版本限制")
                
        except PackageNotFoundError:
            print(f"❌ {req.name:20s} | 状态: 未安装       | (将执行安装)")
            to_install.append(req_str)

    # 3. 处理安装逻辑
    if not to_install:
        print("\n🎉 恭喜！当前环境已完美满足所有版本要求，无需任何修改。")
        return

    print(f"\n⚠️ 发现 {len(to_install)} 个包不符合要求，开始自动安装与修复...\n")
    
    for req_str in to_install:
        print(f">>> 正在安装: {req_str}")
        cmd = [sys.executable, "-m", "pip", "install", req_str]
        
        # 针对特定版本的 PyTorch 增加官方 CUDA 专属下载源
        if "+cu118" in req_str:
            cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu118"])
            
        # 针对 PyG (torch_geometric 相关) 增加对应的 wheels 预编译源，避免本地漫长编译报错
        if any(x in req_str for x in["torch_cluster", "torch_scatter", "torch_sparse"]):
            cmd.extend(["-f", "https://data.pyg.org/whl/torch-2.2.1+cu118.html"])
            
        try:
            subprocess.check_call(cmd)
            print(f"✔️ {req_str} 安装修复成功！\n")
        except subprocess.CalledProcessError:
            print(f"❌ {req_str} 安装失败，请检查网络或对应包名是否有误。\n")

    print("="*50)
    print("✨ 所有不符要求的依赖均已执行更新！建议再次运行本脚本复检。")

if __name__ == "__main__":
    check_and_install()