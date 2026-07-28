# 重构前后一致性检查报告

检查日期：2026-07-28

目标环境：WSL Ubuntu-18.04 / Conda `cgm310`

## 结论

本次补齐的 7 个参考实验在只归一化运行接口差异后，源文件与重构入口的
SHA-256 全部一致。既有 task 脚本只把固定 `use_EDM` 改为可选环境覆盖，
未设置 `DEMO_USE_EDM` 时仍使用原默认值。核心方法文件和目录没有修改。

## 参考脚本审计

运行命令：

```bash
python -m demo_runtime.consistency \
  --reference-root ../GeoLDM
```

| 论文位置 | 参考/本地关系 | 结果 |
|---|---|---|
| 表 2 EGD w/o CO | 新增本地入口 | match |
| 表 2 EGD w/o MT | 新增本地入口 | match |
| 表 3 Top-N | 已有入口，重复命令 | match |
| 表 3 SAES w/o CO | 新增本地入口；内部标签 `SPEA2_woCO` 保留 | match |
| 表 3 SAES w/o MT | 新增本地入口；内部标签 `SPEA2_woMT` 保留 | match |
| 表 5 Only PC/Only-C | 参考文件名为 `DEMO_SAES`，本地使用清晰别名 | match |
| 表 5 DEMO w/o PB | 新增本地入口 | match |

审计仅归一化：

1. 由启动器接管的 GPU 可见性；
2. 保留旧默认值的 `DEMO_USE_EDM` 环境适配；
3. 重复 Top-N 参考脚本中未使用的优化器初始化；
4. CRLF/LF 换行差异。

## 核心方法保护

以下核心位置的 `git diff --name-only` 为空：

```text
evo.py
MOEA/
egnn/
equivariant_diffusion/
qm9/
```

既有 15 个 task 入口的算法体未改动；每个差异都只有：

```python
use_EDM = <旧布尔默认值>
```

变为：

```python
use_EDM = env_bool("DEMO_USE_EDM", default=<同一个旧布尔默认值>)
```

## 目标环境验证

在 WSL Ubuntu-18.04 的 `cgm310` 中完成：

- 15 项单元/回归测试通过，包含 JSONL、完成状态、汇总表和绘图回归；
- 24 个改动或新增 Python 入口通过 `py_compile`；
- 7 个 `scripts/*.sh` 通过 `bash -n`；
- `run_all_revision.sh --dry-run` 成功展开：
  - `qm9_single`：8 条；
  - `qm9_mop`：5 条；
  - `docking_mop`：3 条；
  - `qm9_struct_cmop`：7 条；
  - `qm9_cmop`：20 个种子。

Dry-run 显示论文表格任务均显式传入 `DEMO_USE_EDM=1`，表 1 的 GeoLDM
对照显式传入 `DEMO_USE_EDM=0`。QM9-CMOP 的指标来源记录为
`egnn_property_predictor`，不调用 xTB 或 DFT。

## 已知名称与重复关系

- 表 1/2/3 的 Top-N SR 和 SE 是同一采样轨迹上的统计点，不重复启动；
- 用户提供的表 3 Top-N 参考文件与已有入口重复；
- 表 5 的 `DEMO_SAES.py` 是 Only-PC/Only-C，只启动一次；
- 表 3 消融的论文名和参考脚本内部 `MOEA_NAME` 不一致，重构保留内部实现并
  在 suite 元数据中记录论文名与实现名。

具体命令见
[`PAPER_EXPERIMENT_COMMANDS_ZH.md`](PAPER_EXPERIMENT_COMMANDS_ZH.md)。
