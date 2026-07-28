# 论文表格与新增实验启动命令

本文档给出 `DEMO_Bioinfo.pdf` 表 1–5、本次补充的消融实验，以及
QM9-CMOP 审稿新增实验的命令映射。命令均在项目根目录执行，目标环境为
WSL `Ubuntu-18.04`、Conda `cgm310`。

所有正式命令均支持：

- `--gpus 0,1`：选择物理 GPU；
- `--skip-completed`：跳过身份相同且已完成的运行；
- `--plot`：根据 JSONL 生成 PNG/PDF；
- `--summarize`：生成 JSONL/CSV/Markdown/LaTeX 汇总表；
- `--task TASK_ID`：只启动指定实验，可重复使用以选择多个任务；
- `--dry-run`：只检查命令展开和 GPU 分配。

## 1. 先查看套件中的准确任务名

```bash
bash scripts/run_suite.sh --suite qm9_single --list-tasks
bash scripts/run_suite.sh --suite qm9_mop --list-tasks
bash scripts/run_suite.sh --suite docking_mop --list-tasks
bash scripts/run_suite.sh --suite qm9_struct_cmop --list-tasks
bash scripts/run_suite.sh --suite qm9_cmop --list-tasks
```

输出会同时显示论文方法名、扩散模型骨干和重复实验说明。

## 2. 论文表 1：QM9 单属性目标匹配

运行表 1 的全部本地方法：

```bash
bash scripts/run_suite.sh \
  --suite qm9_single \
  --task single_property_egd_geoldm \
  --task single_property_egd_edm \
  --task single_property_topn_geoldm \
  --task single_property_topn_edm \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

对应关系：

| 论文方法 | 任务 ID | 骨干 |
|---|---|---|
| EGD + GeoLDM | `single_property_egd_geoldm` | GeoLDM |
| EGD + EDM | `single_property_egd_edm` | EDM |
| GeoLDM + Top-N | `single_property_topn_geoldm` | GeoLDM |
| EDM + Top-N | `single_property_topn_edm` | EDM |

**重复说明：** 表中的 SR 和 SE 是同一次 Top-N 采样轨迹上的两个检查点，
不是两条独立命令。每种骨干只启动一次 Top-N。

## 3. 论文表 2：QM9 多属性目标匹配及消融

运行表 2 的全部本地方法：

```bash
bash scripts/run_suite.sh \
  --suite qm9_single \
  --task multi_property_target_egd \
  --task multi_property_target_egd_wo_co \
  --task multi_property_target_egd_wo_mt \
  --task multi_property_target_topn \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

只运行本次补齐的两个消融：

```bash
bash scripts/run_suite.sh \
  --suite qm9_single \
  --task multi_property_target_egd_wo_co \
  --task multi_property_target_egd_wo_mt \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

对应参考文件：

| 论文方法 | 任务 ID | 本地入口 |
|---|---|---|
| EGD w/o CO | `multi_property_target_egd_wo_co` | `task2_target_matching_MAE_woCO.py` |
| EGD w/o MT | `multi_property_target_egd_wo_mt` | `task2_target_matching_MAE_woMT.py` |

**重复说明：** 表 2 的 Top-N SR/SE 同样来自一条采样轨迹，只启动
`multi_property_target_topn` 一次。

## 4. 论文表 3：QM9 无约束多目标优化

运行表 3 的全部本地方法：

```bash
bash scripts/run_suite.sh \
  --suite qm9_mop \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

只运行本次补齐的两个消融：

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

| 论文方法 | 任务 ID | 本地入口 |
|---|---|---|
| SPEA2 | `qm9_mop_spea2` | `task4_dual_pareto_optimize.py` |
| SAES/Ours | `qm9_mop_demo` | `task4_dual_pareto_optimize_DEMO.py` |
| SAES w/o CO | `qm9_mop_saes_wo_co` | `task4_dual_pareto_optimize_woCO.py` |
| SAES w/o MT | `qm9_mop_saes_wo_mt` | `task4_dual_pareto_optimize_woMT.py` |
| Top-N | `qm9_mop_topn` | `task4_dual_pareto_optimize_topn.py` |

**名称说明：** 参考脚本的内部 `MOEA_NAME` 是 `SPEA2_woCO` 和
`SPEA2_woMT`，而论文列名是 “SAES w/o CO/MT”。重构保留参考脚本内部
实现，只在套件元数据中使用论文名称。

**重复说明：** 用户提供的
`GeoLDM/task4_dual_pareto_optimize_topn.py` 与项目中的 Top-N 实验一致；
没有再复制出第二条命令。表 3 的 Top-N SR/SE 也只需一次启动。

## 5. 论文表 4：Docking 多目标优化

```bash
bash scripts/run_suite.sh \
  --suite docking_mop \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

| 论文方法 | 任务 ID |
|---|---|
| SPEA2 | `docking_spea2` |
| SAES/Ours | `docking_demo` |
| Top-N | `docking_topn` |

论文表 4 的本地入口均显式使用 EDM drugs checkpoint。

## 6. 论文表 5：带结构约束的多目标优化

运行表 5 的全部本地方法：

```bash
bash scripts/run_suite.sh \
  --suite qm9_struct_cmop \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

只运行本次补齐的消融：

```bash
bash scripts/run_suite.sh \
  --suite qm9_struct_cmop \
  --task struct_demo_wo_pb \
  --task struct_only_pc \
  --gpus 0,1 \
  --skip-completed \
  --plot \
  --summarize
```

| 论文方法 | 任务 ID | 本地入口 |
|---|---|---|
| SPEA2 | `struct_spea2` | `task8_dual_pareto_optimize_multi_patt1.py` |
| CCMO | `struct_ccmo` | `task8_dual_pareto_optimize_multi_patt1_CCMO.py` |
| CMOEA-CD | `struct_cmoeacd` | `task8_dual_pareto_optimize_multi_patt1_CMOEACD.py` |
| Ours | `struct_demo` | `task8_dual_pareto_optimize_multi_patt1_DEMO.py` |
| Ours w/o PB | `struct_demo_wo_pb` | `task8_dual_pareto_optimize_multi_patt1_DEMO_woB.py` |
| Only PC / Only-C | `struct_only_pc` | `task8_dual_pareto_optimize_multi_patt1_DEMO_onlyC.py` |
| Top-N | `struct_topn` | `task8_dual_pareto_optimize_multi_patt1_topn.py` |

**别名/重复说明：** 参考文件
`task8_dual_pareto_optimize_multi_patt1_DEMO_SAES.py` 就是论文表 5 的
Only-PC/Only-C 消融，已重命名为清晰的本地入口。它们是同一个实验，只
启动 `struct_only_pc` 一次。

## 7. 一键运行论文表格

表 1–5 的所有本地实验：

```bash
bash scripts/run_paper_tables.sh \
  --gpus 0,1,2,3 \
  --results-root results/paper_tables
```

该脚本依次运行 `qm9_single`、`qm9_mop`、`docking_mop` 和
`qm9_struct_cmop`，自动跳过已完成任务并生成图和汇总表。

## 8. 审稿新增：可自定义 QM9-CMOP

先验证属性、目标和约束配置；只报告 EGNN 属性预测器，不调用 xTB/DFT：

```bash
source scripts/_activate_env.sh
python -m demo_runtime.qm9_cmop \
  --problem-config configs/problems/qm9_frontier_alignment.json \
  --validate-config
```

正式启动 20 个种子：

```bash
bash scripts/run_reviewer_additions.sh \
  --gpus 0,1,2,3 \
  --results-root results/reviewer_additions
```

也可以直接运行套件：

```bash
bash scripts/run_suite.sh \
  --suite qm9_cmop \
  --gpus 0,1,2,3 \
  --skip-completed \
  --plot \
  --summarize
```

自定义问题时复制
`configs/problems/qm9_frontier_alignment.json`，并在自定义 suite 的
`identity_files` 中登记新配置路径。

## 9. 一键运行本次已落实的全部实验

```bash
bash scripts/run_all_revision.sh \
  --gpus 0,1,2,3 \
  --results-root results/revision
```

这条命令等价于“论文表 1–5 的本地方法 + 可自定义 QM9-CMOP”。可加
`--dry-run` 先检查，不加载模型：

```bash
bash scripts/run_all_revision.sh \
  --gpus 0,1 \
  --results-root results/revision_dry_run \
  --dry-run
```

## 10. 重构前后一致性检查

在参考文件仍位于相邻 `GeoLDM` 目录时：

```bash
source scripts/_activate_env.sh
python -m demo_runtime.consistency \
  --reference-root ../GeoLDM
```

检查会忽略且仅忽略以下运行接口差异：

- 删除脚本内写死的 `CUDA_VISIBLE_DEVICES`；
- 用 `DEMO_USE_EDM` 选择骨干，但未设置变量时保留旧默认值；
- 原 Top-N 脚本中未使用的优化器初始化。

其余内容按标准化后的 SHA-256 比较；任一 `mismatch` 都会返回非零退出码。
核心方法目录 `evo.py`、`MOEA/`、`egnn/` 和
`equivariant_diffusion/` 不在本次消融迁移的修改范围内。

## 11. 尚未伪造为命令的实验

下列审稿实验没有可直接复用的实现或必要外部程序，因此本次没有用占位命令
冒充已完成：

- cEDM 等外部方法基线；
- 继承/噪声、2D/3D 距离、三种群迁移等新的机制诊断；
- scaffold diversity 和 Docking PoseBusters/相互作用分析；
- 需要新的统一预算定义才能严格比较的 HV(D)/HV(100K)。

这些实验需要先确定数据、预算、指标定义和算法入口，再加入独立 suite。
当前 `run_all_revision.sh` 只覆盖已有实现且已经过接口标准化的实验。
