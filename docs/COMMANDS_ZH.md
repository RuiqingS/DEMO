# DEMO 重构版命令手册

本手册适用于 WSL `Ubuntu-18.04` 和 Conda 环境 `cgm310`。以下命令均在
项目根目录执行。`scripts/*.sh` 会自动寻找 Conda 并激活 `cgm310`，通常
不需要预先执行 `conda activate`。

## 1. 进入环境与项目

从 Windows PowerShell 进入指定 WSL：

```powershell
wsl.exe -d Ubuntu-18.04
```

在 WSL 中进入项目目录：

```bash
cd /path/to/DEMO_re
```

检查系统和 Conda 环境：

```bash
lsb_release -a
source scripts/_activate_env.sh
python --version
conda info --envs
```

如果需要临时使用另一个环境名：

```bash
DEMO_CONDA_ENV=my_env bash scripts/run_suite.sh \
  --suite qm9_cmop \
  --gpus 0 \
  --dry-run
```

## 2. 可用的一键实验套件

| 套件名 | 任务类型 | 主要入口 |
|---|---|---|
| `qm9_single` | 单目标/目标匹配 | 原有 task2 脚本 |
| `qm9_mop` | QM9 无约束多目标 | 原有 task4 脚本 |
| `qm9_cmop` | 可自定义 QM9 约束多目标 | `demo_runtime.qm9_cmop` |
| `qm9_struct_cmop` | 带结构约束的多目标 | 原有 task8 脚本 |
| `docking_mop` | Docking 多目标 | 原有 task5 脚本 |
| `reviewer_noise_inheritance` | 固定噪声—遗传继承曲线 | QM9-CMOP 可选算子接口 |
| `reviewer_scheduler` | 噪声调度器机制 | QM9-CMOP 可选调度器接口 |
| `reviewer_operators` | 算子/选择消融（不含 PAES） | QM9-CMOP 可选算子接口 |
| `reviewer_distance` | SAES 2D/3D 距离消融 | QM9-CMOP 可选距离接口 |
| `reviewer_docking_validation` | PoseBusters、力场与 QVina 可信度 | `demo_runtime.docking_validation` |
| `reviewer_three_population` | A/B/C 谱系与迁移 | task8 的只读审计接口 |
| `reviewer_protocol` | 先验、HV 预算、成本与敏感性 | QM9-CMOP 协议接口 |

所有套件配置位于 `configs/suites/`。

论文表 1–5、补齐消融、重复命令和 QM9-CMOP 的逐项对应关系见
[`PAPER_EXPERIMENT_COMMANDS_ZH.md`](PAPER_EXPERIMENT_COMMANDS_ZH.md)。

## 3. 运行前检查

先验证命令展开和 GPU 分配，不加载模型、不启动正式实验：

```bash
bash scripts/run_suite.sh \
  --suite qm9_cmop \
  --gpus 0,1 \
  --skip-completed \
  --dry-run
```

验证自定义 QM9-CMOP 问题配置：

```bash
source scripts/_activate_env.sh
python -m demo_runtime.qm9_cmop \
  --problem-config configs/problems/qm9_frontier_alignment.json \
  --validate-config
```

输出会列出问题名、目标、约束和所需 EGNN 属性预测器。此命令不加载扩散
模型，不调用 xTB 或 DFT。

运行自动化回归测试：

```bash
source scripts/_activate_env.sh
DEMO_TEST_PLOTTING=1 python -m unittest discover -s tests -v
```

检查 Bash 脚本语法：

```bash
bash -n scripts/_activate_env.sh
bash -n scripts/run_suite.sh
bash -n scripts/run_all_revision.sh
bash -n scripts/run_paper_tables.sh
bash -n scripts/run_reviewer_additions.sh
bash -n scripts/plot_suite.sh
bash -n scripts/summarize_suite.sh
```

## 4. 启动单类实验

推荐的完整命令：

```bash
bash scripts/run_suite.sh \
  --suite qm9_cmop \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

替换 `--suite` 即可运行其他任务：

```bash
bash scripts/run_suite.sh --suite qm9_single --gpus 0,1 --skip-completed
bash scripts/run_suite.sh --suite qm9_mop --gpus 0,1 --skip-completed
bash scripts/run_suite.sh --suite qm9_struct_cmop --gpus 0,1 --skip-completed
bash scripts/run_suite.sh --suite docking_mop --gpus 0,1 --skip-completed
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--suite NAME` | 套件别名，或一个套件 JSON 文件路径 |
| `--gpus 0,1` | 可使用的物理 GPU 编号 |
| `--skip-completed` | 跳过身份完全相同且已成功结束的任务，默认开启 |
| `--max-parallel-per-gpu N` | 每张 GPU 同时运行的进程数，默认 `1` |
| `--results-root PATH` | 结果根目录，默认 `results` |
| `--dry-run` | 只展示任务和 GPU 分配，不运行模型 |
| `--task TASK_ID` | 只运行指定任务；可重复传入以选择多个任务 |
| `--list-tasks` | 列出任务 ID、论文方法、骨干和重复说明后退出 |
| `--plot` | 实验结束后从 JSONL 生成图 |
| `--summarize` | 实验结束后生成汇总表 |

先查看一个套件中的任务：

```bash
bash scripts/run_suite.sh --suite qm9_mop --list-tasks
```

只运行表 3 的两个消融：

```bash
bash scripts/run_suite.sh \
  --suite qm9_mop \
  --task qm9_mop_saes_wo_co \
  --task qm9_mop_saes_wo_mt \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

单卡运行：

```bash
bash scripts/run_suite.sh \
  --suite qm9_cmop \
  --gpus 2 \
  --skip-completed
```

指定结果目录：

```bash
bash scripts/run_suite.sh \
  --suite qm9_mop \
  --gpus 0,1 \
  --results-root results/reviewer_revision \
  --skip-completed
```

同一张 GPU 并行两个任务（仅在显存足够时使用）：

```bash
bash scripts/run_suite.sh \
  --suite qm9_single \
  --gpus 0,1 \
  --max-parallel-per-gpu 2 \
  --skip-completed
```

启动器会在子进程导入 PyTorch 前设置 `CUDA_VISIBLE_DEVICES`。不要再在 task
脚本中写死 GPU 编号。

## 5. 启动全部修改实验

只运行论文表 1–5 的本地实验：

```bash
bash scripts/run_paper_tables.sh --gpus 0,1,2,3
```

只运行审稿意见新增的 8 组实验：

```bash
bash scripts/run_reviewer_additions.sh --gpus 0,1,2,3
```

按顺序运行论文表格和全部审稿新增实验，并在每类结束后绘图和汇总：

```bash
bash scripts/run_all_revision.sh --gpus 0,1,2,3
```

使用自定义结果根目录：

```bash
bash scripts/run_all_revision.sh \
  --gpus 0,1,2,3 \
  --results-root results/reviewer_revision
```

只检查全部套件的命令展开：

```bash
bash scripts/run_all_revision.sh \
  --gpus 0,1 \
  --results-root results/dry_run \
  --dry-run
```

`run_all_revision.sh` 固定传递 `--skip-completed`，因此中断后可直接重复同一
命令。只有 `.complete` 中的 run ID 与当前任务完全一致，且
`events.jsonl` 最后一条是 `run_end/status=completed` 时才会跳过。

## 6. 自定义 QM9-CMOP

复制示例问题：

```bash
cp configs/problems/qm9_frontier_alignment.json \
  configs/problems/my_qm9_cmop.json
```

编辑 `my_qm9_cmop.json` 中的：

- `objectives`：属性、`min`/`max` 方向、单位和固定 HV 边界；
- `derived_properties`：`mean`、`sum` 或 `difference`；
- `constraints`：不等式阈值，或等式目标、容差和尺度；
- `population_size`、`generations`、节点数范围；
- `noise` 和 `runtime.selection`。

直接验证新问题：

```bash
python -m demo_runtime.qm9_cmop \
  --problem-config configs/problems/my_qm9_cmop.json \
  --validate-config
```

直接运行单个种子：

```bash
python -m demo_runtime.qm9_cmop \
  --problem-config configs/problems/my_qm9_cmop.json \
  --seed 0 \
  --output results/manual_my_qm9_cmop_seed0
```

要使用 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 python -m demo_runtime.qm9_cmop \
  --problem-config configs/problems/my_qm9_cmop.json \
  --seed 0 \
  --output results/manual_my_qm9_cmop_seed0
```

批量运行时，复制 `configs/suites/qm9_cmop.json`，把 `--problem-config`
后的路径改成新文件，并把同一路径写入 `identity_files`：

```json
{
  "args": [
    "--problem-config",
    "configs/problems/my_qm9_cmop.json",
    "--seed",
    "{seed}"
  ],
  "identity_files": [
    "configs/problems/my_qm9_cmop.json"
  ]
}
```

随后把该套件 JSON 路径直接传给启动器：

```bash
bash scripts/run_suite.sh \
  --suite configs/suites/my_qm9_cmop.json \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

`identity_files` 的文件内容哈希会参与 run ID。修改目标、约束或阈值后，
新任务不会被旧结果错误跳过。

QM9-CMOP 只使用仓库中的 EGNN 属性预测器，指标元数据记录为
`egnn_property_predictor`；验证流程不调用 xTB 或 DFT。

QM9-CMOP 的审稿实验参数均为可选参数；不传时保留原来的 full
crossover/mutation、Validity×Atom Stability 自适应调度和原 SAES 混合距离：

```text
--operator-mode full|mutation_only|crossover_only|spea2|topn
--scheduler-mode fixed|adaptive_validity|adaptive_inheritance
--fixed-noise 0..1000
--distance-mode objective|morgan|usrcat|mixed|raw_coords
--evaluation-budget N
--training-reference qm9/temp/qm9_smiles.pickle
--population-size N
--min-distance FLOAT
--noise-step N
```

`reviewer_operators` 明确不包含 PAES，命令行的 `--operator-mode` 也不接受
`paes`。

## 7. 审稿新增 8 组实验

先只查看全部任务，不启动模型：

```bash
for suite in \
  reviewer_noise_inheritance \
  reviewer_scheduler \
  reviewer_operators \
  reviewer_distance \
  qm9_cmop \
  reviewer_docking_validation \
  reviewer_three_population \
  reviewer_protocol; do
  bash scripts/run_suite.sh --suite "$suite" --list-tasks
done
```

逐组正式启动：

```bash
bash scripts/run_suite.sh --suite reviewer_noise_inheritance --gpus 0,1 --skip-completed --plot --summarize
bash scripts/run_suite.sh --suite reviewer_scheduler --gpus 0,1 --skip-completed --plot --summarize
bash scripts/run_suite.sh --suite reviewer_operators --gpus 0,1 --skip-completed --plot --summarize
bash scripts/run_suite.sh --suite reviewer_distance --gpus 0,1 --skip-completed --plot --summarize
bash scripts/run_suite.sh --suite qm9_cmop --gpus 0,1 --skip-completed --plot --summarize
bash scripts/run_suite.sh --suite reviewer_docking_validation --gpus 0 --skip-completed --plot --summarize
bash scripts/run_suite.sh --suite reviewer_three_population --gpus 0,1 --skip-completed --plot --summarize
bash scripts/run_suite.sh --suite reviewer_protocol --gpus 0,1 --skip-completed --plot --summarize
```

一次启动全部 8 组：

```bash
bash scripts/run_reviewer_additions.sh \
  --gpus 0,1,2,3 \
  --results-root results/reviewer_additions
```

`protocol_hv_100k` 是高成本任务，建议先独立 dry-run，再单独启动：

```bash
bash scripts/run_suite.sh \
  --suite reviewer_protocol \
  --task protocol_hv_100k \
  --gpus 0,1,2,3 \
  --skip-completed \
  --dry-run

bash scripts/run_suite.sh \
  --suite reviewer_protocol \
  --task protocol_hv_100k \
  --gpus 0,1,2,3 \
  --skip-completed \
  --plot \
  --summarize
```

Docking 可信度验证使用 PoseBusters、MMFF94/UFF、Open Babel 和 QVina，
报告松弛 RMSD、应变释放、重打分一致性、相互作用分类和结合模式多样性；
不调用 xTB 或 DFT。

## 8. 单独重新绘图

绘图只读取已完成任务的 `metrics.jsonl` 和 `population.jsonl`，不会重新
运行模型：

```bash
bash scripts/plot_suite.sh results/qm9_cmop
```

常见输出：

```text
results/qm9_cmop/plots/<problem>/<metric>.png
results/qm9_cmop/plots/<problem>/<metric>.pdf
results/qm9_cmop/plots/<problem>/final_pareto_*.png
results/qm9_cmop/plots/<problem>/final_pareto_*.pdf
```

曲线展示跨已完成运行的均值和 95% 置信区间；二目标和三目标任务会生成
最终经验 Pareto 图。

## 9. 单独生成论文汇总表

```bash
bash scripts/summarize_suite.sh \
  results/qm9_cmop \
  --suite-config configs/suites/qm9_cmop.json
```

输出位于：

```text
results/qm9_cmop/tables/summary.jsonl
results/qm9_cmop/tables/summary.csv
results/qm9_cmop/tables/table_<metric>.md
results/qm9_cmop/tables/table_<metric>.csv
results/qm9_cmop/tables/table_<metric>.tex
```

表格报告均值、标准差、95% 置信区间、中位数、范围和完成运行数 `n`。
LaTeX 表格按套件中的指标方向将最优值加粗、次优值加下划线。未完成任务
不会用最后一代结果补齐。

## 10. 查看 JSONL 与运行状态

单次运行目录结构：

```text
results/<suite>/<problem>/<method>/seed_<seed>_<hash>/
  manifest.json
  metrics.jsonl
  population.jsonl
  lineage.jsonl
  inheritance.jsonl
  transfers.jsonl
  docking_validation.jsonl
  events.jsonl
  stdout.log
  stderr.log
  .complete
```

查看运行身份和实际命令：

```bash
python -m json.tool \
  results/<suite>/<problem>/<method>/seed_<seed>_<hash>/manifest.json
```

查看最新指标：

```bash
tail -n 1 \
  results/<suite>/<problem>/<method>/seed_<seed>_<hash>/metrics.jsonl
```

查看结束状态：

```bash
tail -n 5 \
  results/<suite>/<problem>/<method>/seed_<seed>_<hash>/events.jsonl
```

查找所有已完成任务：

```bash
find results -name .complete -print
```

查找错误日志：

```bash
find results -name stderr.log -size +0 -print
```

## 11. 常见问题

### 找不到 Conda

脚本依次查找 PATH 中的 `conda`、`CONDA_EXE`，以及：

```text
~/miniconda3
~/anaconda3
~/miniforge3
```

如果 Conda 安装在其他位置，先手动初始化：

```bash
source /custom/conda/etc/profile.d/conda.sh
conda activate cgm310
bash scripts/run_suite.sh --suite qm9_cmop --gpus 0 --dry-run
```

### GPU 编号无效

先检查：

```bash
nvidia-smi
```

然后把实际编号传给 `--gpus`。子进程内部看到的设备通常从 `cuda:0`
重新编号，这是 `CUDA_VISIBLE_DEVICES` 的正常行为。

### 任务中断

修复问题后重复原命令即可：

```bash
bash scripts/run_suite.sh \
  --suite qm9_cmop \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

已完成任务会跳过，失败或中断的任务会重新执行。具体错误查看对应运行目录
的 `stderr.log`。

### 修改配置后仍想确认是否会重跑

先执行：

```bash
bash scripts/run_suite.sh \
  --suite configs/suites/my_qm9_cmop.json \
  --gpus 0,1 \
  --skip-completed \
  --dry-run
```

确保自定义问题 JSON 已列在 `identity_files`。问题文件哈希、任务配置、种子、
检查点哈希和 Git commit 都参与运行身份计算。
