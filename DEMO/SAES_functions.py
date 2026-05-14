from utilis import *
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher

def calculate_crystal_similarity(dict_a: dict, dict_b: dict) -> dict:
    """
    计算两个晶体结构的相似程度，并返回详细的对比指标。
    非常适合用于进化算法中的“重复个体剔除”或“拥挤度计算(Crowding Distance)”。

    Args:
        dict_a: 第一个分子的字典
        dict_b: 第二个分子的字典

    Returns:
        包含相似度信息的字典
    """
    try:
        struct_a = dict_to_structure(dict_a)
        struct_b = dict_to_structure(dict_b)

        # 快速前置检查：如果原子总数或化学组分不同，直接判定为不相似
        if struct_a.composition != struct_b.composition:
            return {
                "is_match": False,
                "rms_distance": 1.0,  # 给一个固定的最大距离
                "similarity_score": 0.0  # 0 表示完全不同
            }

        # 初始化结构匹配器
        # ltol: 晶格长度容差 (fraction)
        # stol: 原子位移容差
        # angle_tol: 角度容差 (度)
        matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

        # 1. 离散判断：它们在拓扑上是否等效（物理上是不是同一种材料？）
        is_match = matcher.fit(struct_a, struct_b)

        # 2. 连续距离判断：均方根位移 (Root Mean Square Displacement, RMSD)
        # get_rms_dist 会自动寻找最佳的平移和映射方式，然后计算原子间距离差
        rms_data = matcher.get_rms_dist(struct_a, struct_b)

        if rms_data is not None:
            # rms_data 是一个元组: (rms_dist, max_dist)
            rms_distance = rms_data[0]
            # 将 RMSD 转换为一个 0 到 1 之间的相似度分数 (RMSD 越小，分数越接近 1)
            similarity_score = np.exp(-rms_distance)
        else:
            # 如果匹配器认为它们差异太大无法映射（即使组分相同）
            rms_distance = 1.0  # 视作最大距离 1.0 埃
            similarity_score = 0.0

    except Exception as e:
        print(f"相似度计算异常: {e}")
        is_match = False
        rms_distance = 1.0
        similarity_score = 0.0

    return {
        "is_match": is_match,  # 布尔值：是否为等效材料
        "rms_distance": rms_distance,  # 连续值：几何均方根位移（埃），越小越相似
        "similarity_score": similarity_score  # 连续值：[0, 1]区间，1代表完全一样
    }




def _is_too_similar(
        sim_info: dict,
        score_threshold: float,
        rms_threshold: float,
) -> bool:
    """
    根据给定阈值判断是否“过于相似”。
    """
    score = float(sim_info.get("similarity_score", 0.0))
    rms = float(sim_info.get("rms_distance", 1.0))
    is_match = bool(sim_info.get("is_match", False))
    return is_match and score >= score_threshold and rms <= rms_threshold


def prune_similar_population(
        population: list[dict],
        similar_score_threshold: float,
        similar_rms_threshold: float,
        very_similar_score_threshold: float,
        very_similar_rms_threshold: float,
) -> list[dict]:
    """
    按适应度从高到低保留种群，去除过于相似的个体。
    """
    # 直接使用 x.get("fitness")，而不是 x["dict"]["fitness"]
    sorted_pop = sorted(population, key=lambda x: x.get("fitness", -999), reverse=True)
    kept: list[dict] = []

    for cand in sorted_pop:
        discard = False
        for old in kept:
            # cand 和 old 直接就是晶体字典
            sim_info = calculate_crystal_similarity(cand, old)
            both_valid = cand.get("is_valid", False) and old.get("is_valid", False)

            if both_valid:
                if _is_too_similar(
                        sim_info,
                        score_threshold=very_similar_score_threshold,
                        rms_threshold=very_similar_rms_threshold,
                ):
                    discard = True
                    break
            else:
                if _is_too_similar(
                        sim_info,
                        score_threshold=similar_score_threshold,
                        rms_threshold=similar_rms_threshold,
                ):
                    discard = True
                    break
        if not discard:
            kept.append(cand)

    return kept


import numpy as np
from SAES_functions import *
from characterize_functions import *

# ================= 1. 动态约束与目标评估模块 =================

def evaluate_constraints(data_dict: dict, constraints: dict) -> dict:
    """根据外部传入的字典动态计算约束违反值"""
    violation = 0.0
    for key, rules in constraints.items():
        val = data_dict.get(key)

        if val is None:
            violation += 100.0
            continue

        weight = rules.get("weight", 1.0)

        # 1. 布尔类型约束
        if type(val) is bool:
            expected = rules.get("expected", True)
            if val != expected:
                violation += weight

        # 2. 离散匹配约束 (白名单/黑名单)
        elif "allowed_only" in rules or "banned" in rules:
            # A. 如果值是一个列表/集合 (例如 element_set)
            if isinstance(val, (list, set, tuple)):
                if "banned" in rules:
                    overlap = set(val).intersection(set(rules["banned"]))
                    if overlap:
                        violation += weight * len(overlap)
                if "allowed_only" in rules:
                    out_of_bounds = set(val) - set(rules["allowed_only"])
                    if out_of_bounds:
                        violation += weight * len(out_of_bounds)

            # B. 【新增】：如果值是一个单独的标量 (例如 空间群 sg_number = 194)
            else:
                if "banned" in rules and val in rules["banned"]:
                    violation += weight
                if "allowed_only" in rules and val not in rules["allowed_only"]:
                    violation += weight

        # 3. 连续数值类型约束 (min/max)
        else:
            if "min" in rules and val < rules["min"]:
                violation += (rules["min"] - val) * weight
            if "max" in rules and val > rules["max"]:
                violation += (val - rules["max"]) * weight

    data_dict["total_violation"] = violation
    data_dict["is_valid"] = bool(violation <= 1e-5)
    return data_dict


# ================= 2. FNDS-CDP 快速非支配排序核心模块 =================

def dominates_cdp(ind1: dict, ind2: dict, objectives: dict) -> bool:
    """
    基于约束支配原则 (CDP) 判断 ind1 是否支配 ind2
    """
    v1 = ind1.get("total_violation", float('inf'))
    v2 = ind2.get("total_violation", float('inf'))

    # 规则 1 & 2: 优先比较约束违反程度 (越小越好)
    # 给定一个极小的容差，防止浮点误差导致的误判
    if v1 < v2 - 1e-5:
        return True
    if v1 > v2 + 1e-5:
        return False

    # 规则 3: 约束违反程度相同 (例如都是 0，即都是合法解)
    # 进行目标空间的 Pareto 支配判断
    better_in_all = True
    strictly_better_in_one = False

    for obj_key, direction in objectives.items():
        val1 = ind1.get(obj_key, float('inf') if direction == 'minimize' else -float('inf'))
        val2 = ind2.get(obj_key, float('inf') if direction == 'minimize' else -float('inf'))

        if direction == "minimize":
            if val1 > val2: better_in_all = False
            if val1 < val2: strictly_better_in_one = True
        elif direction == "maximize":
            if val1 < val2: better_in_all = False
            if val1 > val2: strictly_better_in_one = True

    return better_in_all and strictly_better_in_one


def fast_non_dominated_sorting(population: list[dict], objectives: dict) -> list[dict]:
    """
    FNDS: 对种群进行快速非支配排序，计算每个个体的 pareto_rank
    """
    fronts = [[]]
    for p in population:
        p["dominated_set"] = []
        p["domination_count"] = 0

        for q in population:
            if dominates_cdp(p, q, objectives):
                p["dominated_set"].append(q)
            elif dominates_cdp(q, p, objectives):
                p["domination_count"] += 1

        if p["domination_count"] == 0:
            p["pareto_rank"] = 1
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p["dominated_set"]:
                q["domination_count"] -= 1
                if q["domination_count"] == 0:
                    q["pareto_rank"] = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    fronts.pop()  # 移除最后一个空列表

    # 清理辅助变量，并将 Rank 映射为兼容剪枝函数的 fitness 代理值
    # Rank 越低 (比如1)，fitness 越高 (比如 -1)
    for ind in population:
        ind.pop("dominated_set", None)
        ind.pop("domination_count", None)
        ind["fitness"] = -ind["pareto_rank"]

    return population


# ================= 3. 环境选择流水线 =================

def evolutionary_environmental_selection(
        combined_population: list[dict],
        constraints: dict,
        objectives: dict,
        pop_size: int,
        similar_score_threshold: float,
        similar_rms_threshold: float,
        very_similar_score_threshold: float,
        very_similar_rms_threshold: float
) -> list[dict]:
    # 1. 直接评估约束 (因为此时 item 就是晶体字典)
    for item in combined_population:
        evaluate_constraints(item, constraints)

    # 2. 快速非支配排序 (基于 CDP)
    fast_non_dominated_sorting(combined_population, objectives)

    # 3. 相似度剪枝
    pruned_pop = prune_similar_population(
        combined_population,
        similar_score_threshold=similar_score_threshold,
        similar_rms_threshold=similar_rms_threshold,
        very_similar_score_threshold=very_similar_score_threshold,
        very_similar_rms_threshold=very_similar_rms_threshold,
    )

    # 4. 精英截断
    return pruned_pop[:pop_size]