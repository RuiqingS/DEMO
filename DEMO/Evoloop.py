import copy
import csv
import sys
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from EGD_functions import *
from SAES_functions import *
from characterize_functions import *
from utilis import *
import torch


import os
import time
import atexit
import signal
import socket
import subprocess
from EMO_frameworks import StandardCDPEngine, TriplePopCMOEAEngine
# ================== 自动化 JACS 微服务管理 (PyCharm 免疫版) ==================
_jacs_server_process = None
_jacs_server_log_file = None


def is_port_in_use(port: int) -> bool:
    """探针：检查指定端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def start_jacs_server():
    """在后台静默启动 JACS 微服务，支持断线重连和复用"""
    global _jacs_server_process, _jacs_server_log_file

    port = 8080

    # 【神级优化】：如果发现端口已经被占用了（比如 PyCharm 暴力重启留下的僵尸）
    # 我们直接复用它！省去了 10 秒的模型加载时间！
    if is_port_in_use(port):
        print(f"⚡ 检测到 JACS 微服务已在端口 {port} 驻留，直接极速复用！")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    kaist_repo_dir = os.path.join(base_dir, "PUCGCNN")
    server_script = os.path.join(kaist_repo_dir, "jacs_server.py")
    pu_python = "/home/ruiqingwsl2404/miniconda3/envs/mattergen/bin/python"

    if not os.path.exists(server_script):
        print(f"⚠️ 警告: 找不到 JACS 服务器脚本 {server_script}，跳过启动。")
        return

    print(f"🚀 正在后台冷启动 JACS 微服务引擎 (需等待几秒加载 100 个模型)...")

    _jacs_server_log_file = open("jacs_server.log", "w")
    _jacs_server_process = subprocess.Popen(
        [pu_python, server_script],
        cwd=kaist_repo_dir,
        stdout=_jacs_server_log_file,
        stderr=subprocess.STDOUT
    )

    # 动态等待，直到端口开放
    max_retries = 15
    for _ in range(max_retries):
        time.sleep(1)
        if is_port_in_use(port):
            print("✅ JACS 微服务启动并驻留内存成功！")
            return

    print("❌ JACS 微服务启动超时，请查看 jacs_server.log 排错。")


def stop_jacs_server(*args):
    """安全清理钩子"""
    global _jacs_server_process, _jacs_server_log_file
    if _jacs_server_process is not None:
        try:
            print("\n🛑 正在安全关闭 JACS 后台微服务...")
            _jacs_server_process.terminate()
            _jacs_server_process.wait(timeout=3)
        except Exception:
            pass  # 忽略关闭时的异常
        finally:
            if _jacs_server_log_file:
                _jacs_server_log_file.close()
            _jacs_server_process = None
            print("✅ JACS 微服务已释放。")


# 1. 注册正常结束的退出钩子
atexit.register(stop_jacs_server)

# 2. 拦截系统终止信号（增强对 PyCharm Stop 按钮的抵抗力）
try:
    signal.signal(signal.SIGTERM, stop_jacs_server)
    signal.signal(signal.SIGINT, stop_jacs_server)
except Exception:
    pass


# ==========================================================


def main():
    # ================== 1. 自动化服务与信号拦截 ==================
    start_jacs_server()
    torch.set_grad_enabled(False)

    stop_requested = False
    stop_signal_count = 0

    def _handle_stop(signum, frame):
        nonlocal stop_requested, stop_signal_count
        stop_signal_count += 1
        if stop_signal_count >= 2:
            print(f"\n[Signal] 连续收到终止信号 {signum}，强行退出。")
            os._exit(130)
        stop_requested = True
        print(f"\n[Signal] 收到信号 {signum}，将在当前代结束后安全停止。")

    try:
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
    except Exception:
        pass

    # ================== 2. 基础路径与超参配置 ==================
    model_path = "/mnt/d/RemoteServer/10.107.11.90/mattergen/checkpoints/mattergen_base"
    cif_dir = "/mnt/d/RemoteServer/10.107.11.90/mattergen/results/2D/"
    output_root = "/mnt/d/RemoteServer/10.107.11.90/mattergen/test_output/2Dresults"

    # ========== Core Evolutionary Parameters ==========
    noise_level = 0.3  # Diffusion timestep for EGD (0.0-1.0)
    # Higher = more aggressive mutation
    n_pops = 64  # Population size
    n_offspring_per_generation = 64  # Number of offspring per generation
    max_generations = 20  # Total generations to evolve
    n_independent_runs = 10  # Number of independent runs with different seeds
    random_seed = 5  # Base random seed
    trajectory_sample_every_k = 5  # Save trajectory snapshots every k generations

    # ========== Structural Similarity Pruning ==========
    # Remove duplicate structures to maintain diversity
    similar_score_threshold = 0.80  # Similarity score threshold (0-1)
    similar_rms_threshold = 0.20  # RMS distance threshold (Angstrom)
    very_similar_score_threshold = 0.80  # Stricter threshold for very similar structures
    very_similar_rms_threshold = 0.08  # Stricter RMS threshold

    # ========== Model Generation Settings ==========
    model_init_batch_size = 64  # Batch size for initial population generation
    model_init_max_attempts = 20  # Max attempts to generate valid structures

    limit_density = load_corruptions_from_model_config(model_path)["limit_density"]

    # ================== 3. 动态约束与目标定义 ==================
    my_constraints = {
        "bandgap_eV": {"min": 0.2, "max": 3.5, "weight": 5.0},
        "exfoliation_energy_meV": {"min": -5.0, "max": 350.0, "weight": 5.0},
        "formation_energy_eV": {"max": 0, "weight": 10.0},
        "cl_score": {"min": 0.50, "weight": 5.0},
        "f_max_eV_A": {"max": 2.0, "weight": 5.0},
        "ehull_eV": {"max": 1.0, "weight": 5.0},
        "num_elements": {"min": 3, "max": 3, "weight": 10.0},
        "element_set": {
            "banned": [
                "Hg", "Tl", "Na", "K", "F" # 有毒
            ],
            # "allowed_only": ["Mo", "W", "V", "Nb", "S", "Se", "Te", "O"],
            "weight": 5.0
        },
        "sg_number": {
            # "allowed_only": [],
            "banned": [1],
            "weight": 1.0
        }
        # "has_piezo_potential": {"expected": True, "weight": 5.0} # 需要时解开
    }

    my_objectives = {
        "ehull_eV": "minimize",
        "dielectric_epsx": "maximize"
    }

    objective_x, objective_y = list(my_objectives.keys())[:2]

    # ================== 4. 独立运行循环 (Runs) ==================
    os.makedirs(output_root, exist_ok=True)
    launch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cif_files = sorted([os.path.join(cif_dir, f) for f in os.listdir(cif_dir) if f.endswith(".cif")])

    for run_idx in range(1, n_independent_runs + 1):
        if stop_requested:
            print("[Main] 检测到停止请求，退出批处理任务。")
            break

        run_seed = random_seed + run_idx - 1
        random.seed(run_seed)
        torch.manual_seed(run_seed)

        # 目录初始化
        run_dir = os.path.join(output_root, f"run_{run_idx:03d}_{launch_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        # 动态决定当前 Run 的初始化模式
        init_mode = "cif_plus_model_fill" if run_idx % 2 == 1 else "cif_only"

        current_settings = {
            "launch_timestamp": launch_timestamp,
            "run_idx": run_idx,
            "run_seed": run_seed,
            "init_mode": init_mode,
            "algorithm_framework": "DEMO",
            "hyperparameters": {
                "n_pops_per_group": n_pops,
                "n_offspring_per_generation": n_offspring_per_generation,
                "max_generations": max_generations,
                "noise_level": noise_level,
                "model_init_batch_size": model_init_batch_size,
                "model_init_max_attempts": model_init_max_attempts
            },
            "pruning_thresholds": {
                "similar_score_threshold": similar_score_threshold,
                "similar_rms_threshold": similar_rms_threshold,
                "very_similar_score_threshold": very_similar_score_threshold,
                "very_similar_rms_threshold": very_similar_rms_threshold
            },
            "constraints": my_constraints,
            "objectives": my_objectives,
            "paths": {
                "model_path": model_path,
                "cif_dir": cif_dir,
                "output_root": output_root
            }
        }

        # 写入 run_settings.json
        settings_path = os.path.join(run_dir, "run_settings.json")
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(current_settings, f, indent=4, ensure_ascii=False)

        print(f"📄 当前实验配置已保存至: {settings_path}")
        trajectory_dir = os.path.join(run_dir, "trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)

        # 【新增】：合格晶体独立保存库
        qualified_dir = os.path.join(run_dir, "qualified_crystals")
        os.makedirs(qualified_dir, exist_ok=True)
        global_qualified_archive = {}  # 用于跨代去重记录合格晶体

        print(f"\n========== Run {run_idx}/{n_independent_runs} (seed={run_seed}) ==========")

        # ---------------- A. 初始种群构建 ----------------
        initial_pool = []
        for cif_path in cif_files:
            name = os.path.basename(cif_path)
            crystal = extract_all_properties(cif_to_dict(cif_path))
            crystal.update({"name": name, "path": cif_path, "meta": {"type": "initial"}})
            initial_pool.append(crystal)
            print_crystal_properties(name, crystal)

        # 交替使用模型补齐策略
        init_mode = "cif_plus_model_fill" if run_idx % 2 == 0 else "cif_only"

        if init_mode == "cif_plus_model_fill" and len(initial_pool) < n_pops:
            need_count = n_pops - len(initial_pool)
            model_init_dir = os.path.join(run_dir, "initial_model_generated")
            print(f"[Init] 需利用扩散模型随机生成 {need_count} 个晶体补齐种群...")

            generated_pool = generate_model_initial_pool(
                need_count=need_count,
                model_out_dir=model_init_dir,
                run_seed=run_seed,
                run_idx=run_idx,
                constraints=my_constraints,
                model_path=model_path,
                model_init_batch_size=model_init_batch_size,
                model_init_max_attempts=model_init_max_attempts,
                extract_properties_fn=extract_all_properties,
                evaluate_constraints_fn=evaluate_constraints,
            )
            initial_pool.extend(generated_pool)

        if len(initial_pool) < 2:
            raise ValueError("初始候选者不足2个，无法启动进化。")

        # ---------------- B. 大脑引擎初始化 ----------------
        emo_engine = TriplePopCMOEAEngine(
            constraints=my_constraints, objectives=my_objectives, pop_size=n_pops,
            similar_score_threshold=similar_score_threshold, similar_rms_threshold=similar_rms_threshold,
            very_similar_score_threshold=very_similar_score_threshold,
            very_similar_rms_threshold=very_similar_rms_threshold
        )
        emo_engine.initialize(initial_pool)

        generation_idx = 0
        sampled_snapshots = []

        # ---------------- C. 进化主循环 ----------------
        while generation_idx < max_generations:
            if stop_requested: break
            generation_idx += 1
            gen_dir = os.path.join(run_dir, f"gen_{generation_idx:03d}")
            os.makedirs(gen_dir, exist_ok=True)

            current_pop = emo_engine.get_current_population()
            print(f"\n[Run {run_idx} | Gen {generation_idx}] 开始, 总子群容量: {len(current_pop)}")

            # 1. Ask: 请求子代变异配置
            pending_denoise, offspring_meta = emo_engine.ask(
                n_offspring=n_offspring_per_generation,
                parent_selection_mode="tournament",
                tournament_k=2,
                noise_level=noise_level,
                model_path=model_path,
                limit_density=limit_density
            )

            # 2. Do: 物理去噪与属性测算
            offspring_population = []
            if pending_denoise:
                denoised_offspring = denoise_batch(pending_denoise, model_path_or_name=model_path)
                for idx, (meta, child) in enumerate(zip(offspring_meta, denoised_offspring)):
                    child = extract_all_properties(child)
                    child_name = f"gen_{generation_idx:03d}_{meta['type']}_{idx:03d}.cif"
                    child_path = os.path.join(gen_dir, child_name)

                    # 原始子代暂存在当代文件夹
                    dict_to_cif(child, child_path)
                    child.update({"name": child_name, "path": child_path, "meta": meta})
                    offspring_population.append(child)

            # 3. Tell: 环境淘汰与种群更新
            emo_engine.tell(offspring_population)
            updated_pop = emo_engine.get_current_population()

            # ================= 4. 全局合格档案与极简报告 =================
            report_rows = []

            for cand in updated_pop:
                # 检查是否为合格晶体
                is_valid = cand.get("is_valid", False)
                if not is_valid:
                    continue  # 【核心需求】：不合格晶体不写入每代报告！

                # 存入全局档案馆 (按名字去重)
                cand_name = cand.get("name", "unknown")
                if cand_name not in global_qualified_archive:
                    global_qualified_archive[cand_name] = cand

                    # 额外把它的 cif 拷贝到专门的黄金合格区！
                    safe_qualified_path = os.path.join(qualified_dir, cand_name)
                    dict_to_cif(cand, safe_qualified_path)

                # 生成 CSV 行数据
                row = {"Pop_Type": cand.get("pop_type", "Unknown"), "Name": cand_name}
                row.update(auto_extract_record(cand))
                report_rows.append(row)

            # 保存每代的纯净版合格报告
            if report_rows:
                csv_fieldnames = ["Pop_Type", "Name"]
                for row in report_rows:
                    for k in row.keys():
                        if k not in csv_fieldnames: csv_fieldnames.append(k)

                with open(os.path.join(gen_dir, "qualified_report.csv"), "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writeheader()
                    writer.writerows(report_rows)

            print(f"✅ 第 {generation_idx} 代发现合格晶体: {len(report_rows)} 个。")

            # 记录轨迹快照
            if generation_idx % trajectory_sample_every_k == 0 or generation_idx == max_generations:
                sampled_snapshots.append({
                    "generation": generation_idx,
                    "population": copy.deepcopy(updated_pop)
                })

            del pending_denoise, offspring_meta, offspring_population
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # ---------------- D. Run 结束总结与全局重新评估 ----------------

        # 1. 终极合格品 Pareto 重新洗牌
        all_qualified_crystals = list(global_qualified_archive.values())
        print(
            f"\n🏆 [Run {run_idx} 终局] 累计搜集到合格晶体: {len(all_qualified_crystals)} 个，正在进行终极非支配排序...")

        final_report_rows = []
        if all_qualified_crystals:
            # 去除原有的 rank，用整体重新排 (由于全部合格，FNDS 将纯拼目标函数)
            fast_non_dominated_sorting(all_qualified_crystals, my_objectives)

            # 按 Pareto Rank 从最牛的开始排
            all_qualified_crystals.sort(key=lambda x: x.get("pareto_rank", 999))

            for rank_idx, cand in enumerate(all_qualified_crystals, start=1):
                row = {"Final_Rank": rank_idx, "Name": cand.get("name")}
                row.update(auto_extract_record(cand))
                final_report_rows.append(row)

            # 保存总榜单
            final_csv_path = os.path.join(run_dir, "FINAL_QUALIFIED_LEADERBOARD.csv")
            csv_fieldnames = ["Final_Rank", "Name"]
            for row in final_report_rows:
                for k in row.keys():
                    if k not in csv_fieldnames: csv_fieldnames.append(k)

            with open(final_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writeheader()
                writer.writerows(final_report_rows)

            print(f"📄 最终精英榜单已生成：{final_csv_path}")

        # 2. 绘制多模态目标空间演化图
        print(f"📈 正在绘制多模态目标空间演化轨迹图...")
        try:
            plot_multimodal_evolution_trajectory(
                sampled_snapshots=sampled_snapshots,
                objective_x=objective_x,
                objective_y=objective_y,
                save_dir=trajectory_dir,
            )
            print("✅ 演化图绘制完毕！")
        except Exception as e:
            print(f"⚠️ 演化图绘制失败: {e}")

        # 3. 绘制最终的精英分布层级图！
        print(f"🌟 正在绘制终极 Pareto 精英分布图...")
        try:
            pareto_plot_path = os.path.join(run_dir, "FINAL_PARETO_FRONTS.png")
            plot_final_elite_pareto(
                final_elite_records=all_qualified_crystals,  # <--- 【极度关键】：传入未经排版的原始字典集合！
                objective_x=objective_x,
                objective_y=objective_y,
                save_path=pareto_plot_path
            )
            print("✅ 终极精英图绘制完毕！")
        except Exception as e:
            print(f"⚠️ 精英图绘制失败: {e}")

        # 4. 保存 Run Summary
        run_summary = {
            "run_idx": run_idx,
            "seed": run_seed,
            "max_generations": max_generations,
            "total_qualified_discovered": len(all_qualified_crystals),
            "objective_x": objective_x,
            "objective_y": objective_y,
            "leaderboard_path": final_csv_path if all_qualified_crystals else "None",
            "pareto_plot_path": pareto_plot_path if all_qualified_crystals else "None"
        }
        with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)

        print(f"🏁 Run {run_idx} 完美收官！")



if __name__ == "__main__":
    main()

