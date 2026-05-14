import random
import copy
from SAES_functions import *
from EGD_functions import *


class BaseEMOEngine:
    """EMO 优化器的基类，所有自定义算法都必须继承并实现这三个方法"""

    def initialize(self, initial_pool: list[dict]):
        raise NotImplementedError

    def ask(self, **kwargs) -> tuple[list, list]:
        raise NotImplementedError

    def tell(self, evaluated_offspring: list[dict]):
        raise NotImplementedError

    def get_current_population(self) -> list[dict]:
        raise NotImplementedError


class StandardCDPEngine(BaseEMOEngine):
    """
    你目前的算法：单种群 + 锦标赛选择 + CDP约束支配 + FNDS快速非支配排序 + 结构剪枝
    """

    def __init__(self, constraints, objectives, pop_size,
                 similar_score_threshold, similar_rms_threshold,
                 very_similar_score_threshold, very_similar_rms_threshold):
        self.constraints = constraints
        self.objectives = objectives
        self.pop_size = pop_size
        self.population = []

        self.sim_score = similar_score_threshold
        self.sim_rms = similar_rms_threshold
        self.v_sim_score = very_similar_score_threshold
        self.v_sim_rms = very_similar_rms_threshold

    def initialize(self, initial_pool: list[dict]):
        """初始化第一代种群并排序"""
        self.population = evolutionary_environmental_selection(
            initial_pool, self.constraints, self.objectives, self.pop_size,
            self.sim_score, self.sim_rms, self.v_sim_score, self.v_sim_rms
        )

    def ask(self, n_offspring, parent_selection_mode, tournament_k, noise_level, model_path, limit_density) -> tuple[
        list, list]:
        """
        大脑告诉主程序：我要怎么交叉变异。
        返回：需要拿去模型去噪的 pending_denoise 列表，以及它们的 meta 信息。
        """
        pending_denoise = []
        offspring_meta = []

        while len(offspring_meta) < n_offspring:
            parent_source = self.population if len(self.population) >= 2 else []  # 如果种群空了(极少发生)需要外部兜底，此处简写

            p1_info, p2_info = select_parents(
                parent_source=parent_source,
                mode=parent_selection_mode,
                tournament_k=tournament_k,
            )

            # 【注意】：这里你需要保证 to_noise_input 在环境里可用，或者直接对 dict 操作
            p1_for_noise = p1_info  # 假设这里直接传字典
            p2_for_noise = p2_info

            noisy_p1 = add_noise(p1_for_noise, t=noise_level, model_path_or_name=model_path)
            noisy_p2 = add_noise(p2_for_noise, t=noise_level, model_path_or_name=model_path)
            noisy_cross = crossover(noisy_p1, noisy_p2, limit_density=limit_density)

            p1_name = p1_info.get("name", "unknown")
            p2_name = p2_info.get("name", "unknown")

            # 加入交叉子代
            cross_idx = len(offspring_meta)
            noisy_cross["name"] = f"crossover_{cross_idx:03d}_noisy.cif"
            pending_denoise.append(copy.deepcopy(noisy_cross))
            offspring_meta.append(
                {"type": "crossover", "parent_1": p1_name, "parent_2": p2_name, "offspring_idx": cross_idx})
            if len(offspring_meta) >= n_offspring: break

            # 加入变异子代 (1:1 比例)
            mut_idx = len(offspring_meta)
            if random.random() < 0.5:
                noisy_mut, mut_parent_name = noisy_p1, p1_name
            else:
                noisy_mut, mut_parent_name = noisy_p2, p2_name

            noisy_mut["name"] = f"mutation_{mut_idx:03d}_noisy.cif"
            pending_denoise.append(copy.deepcopy(noisy_mut))
            offspring_meta.append(
                {"type": "mutation", "parent_1": mut_parent_name, "parent_2": "", "offspring_idx": mut_idx})

        return pending_denoise, offspring_meta

    def tell(self, evaluated_offspring: list[dict]):
        """
        主程序把算好带隙、能量的子代还给大脑。
        大脑执行 CDP 排序和剪枝，更新内部种群状态。
        """
        merged_population = self.population + evaluated_offspring
        self.population = evolutionary_environmental_selection(
            merged_population, self.constraints, self.objectives, self.pop_size,
            self.sim_score, self.sim_rms, self.v_sim_score, self.v_sim_rms
        )

    def get_current_population(self) -> list[dict]:
        return self.population


# ======================= 辅助算子：SPEA2 环境选择 =======================
def spea2_selection(population: list[dict], objectives: dict, pop_size: int) -> list[dict]:
    """
    针对可行解的 SPEA2 环境选择：结合了支配强度(Raw Fitness)与结构密度(Density)。
    """
    n = len(population)
    if n <= pop_size:
        return population

    # 1. 确定支配关系
    dominators = {i: [] for i in range(n)}
    dominatees = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j: continue
            # SPEA2 在这里只比较目标，因为进来的都是可行解 (违反度为0)
            # 为了复用之前写的 dominates_cdp，我们直接调用
            # (注意: 这里假设 dominates_cdp 内部是从字典里取值的)
            better_in_all = True
            strictly_better = False
            for obj_key, direction in objectives.items():
                v1 = population[i].get(obj_key, float('inf') if direction == 'minimize' else -float('inf'))
                v2 = population[j].get(obj_key, float('inf') if direction == 'minimize' else -float('inf'))
                if direction == "minimize":
                    if v1 > v2: better_in_all = False
                    if v1 < v2: strictly_better = True
                else:
                    if v1 < v2: better_in_all = False
                    if v1 > v2: strictly_better = True

            if better_in_all and strictly_better:
                dominatees[i].append(j)  # i 支配 j
                dominators[j].append(i)  # j 被 i 支配

    # 2. 计算 Raw Fitness (R(i))
    strength = {i: len(dominatees[i]) for i in range(n)}
    raw_fitness = {i: sum(strength[j] for j in dominators[i]) for i in range(n)}

    # 3. 计算结构密度 Density (D(i))
    # 这里我们使用真实晶体相似度来计算距离：距离 = 1.0 - similarity_score
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = calculate_crystal_similarity(population[i], population[j])["similarity_score"]
            dist = 1.0 - sim_score
            distances[i, j] = dist
            distances[j, i] = dist

    # SPEA2 取第 k 近的距离，通常 k = sqrt(n)
    k = int(np.sqrt(n))
    density = {}
    for i in range(n):
        sorted_dist = np.sort(distances[i])
        kth_dist = sorted_dist[k]
        density[i] = 1.0 / (kth_dist + 2.0)

    # 4. 综合 Fitness 越小越好 (非支配解的 R(i)=0)
    for i in range(n):
        population[i]["spea2_fitness"] = raw_fitness[i] + density[i]
        # 为了兼容外部的剪枝代码，映射成统一的 fitness (越大越好，所以取反)
        population[i]["fitness"] = -population[i]["spea2_fitness"]

    # 5. 环境截断 (其实 SPEA2 还有 Archive 维护逻辑，这里简化为截断)
    sorted_pop = sorted(population, key=lambda x: x["spea2_fitness"])
    return sorted_pop[:pop_size]


def calculate_objective_crowding_distance(population: list[dict], objectives: dict):
    """计算目标空间中的拥挤距离 (修复了 inf 叠加导致无法区分的 BUG)"""
    n = len(population)
    for p in population:
        p["crowding_distance"] = 0.0

    if n <= 2:
        for p in population:
            p["crowding_distance"] = 1e6  # 使用 100万 代替 inf
        return

    for obj_key, direction in objectives.items():
        # 安全取值辅助函数 (处理缺失值或极值)
        def get_val(ind):
            v = ind.get(obj_key)
            if v is None or v == float('inf') or v == -float('inf') or v == 999.0:
                return float('inf') if direction == 'minimize' else -float('inf')
            return float(v)

        # 按当前目标排序
        population.sort(key=lambda x: get_val(x))

        # 边界点给予极大常数奖励 (可叠加)
        population[0]["crowding_distance"] += 1e6
        population[-1]["crowding_distance"] += 1e6

        obj_min = get_val(population[0])
        obj_max = get_val(population[-1])

        # 异常数据保护
        if obj_min in (float('inf'), -float('inf')) or obj_max in (float('inf'), -float('inf')):
            continue

        scale = abs(obj_max - obj_min)
        if scale == 0:
            scale = 1.0

        # 累加内部点的归一化距离
        for i in range(1, n - 1):
            val_next = get_val(population[i + 1])
            val_prev = get_val(population[i - 1])

            if val_next in (float('inf'), -float('inf')) or val_prev in (float('inf'), -float('inf')):
                continue

            distance = abs(val_next - val_prev) / scale
            population[i]["crowding_distance"] += distance

# ======================= 三种群 CMOEA 引擎 =======================
class TriplePopCMOEAEngine(BaseEMOEngine):
    """
    CMOEA 三种群协同框架：
    - Pop C (主种群): 仅收可行解，使用 SPEA2 优化
    - Pop B (辅种群): 仅收不可行解，使用 CDP 引导走向可行域
    - Pop A (辅种群): 无视约束，使用 FNDS 优化(目标+远离Pop C)，保持结构多样性
    """

    def __init__(self, constraints, objectives, pop_size,
                 similar_score_threshold, similar_rms_threshold,
                 very_similar_score_threshold, very_similar_rms_threshold):
        self.constraints = constraints
        self.objectives = objectives
        # 每个种群分配 1/3 的名额，或者 A,B,C 都保持 pop_size 的上限
        self.max_size_c = pop_size
        self.max_size_b = pop_size
        self.max_size_a = pop_size

        self.pop_c = []  # Feasible
        self.pop_b = []  # Infeasible
        self.pop_a = []  # Diverse Unconstrained

        self.prune_params = {
            "similar_score_threshold": similar_score_threshold,
            "similar_rms_threshold": similar_rms_threshold,
            "very_similar_score_threshold": very_similar_score_threshold,
            "very_similar_rms_threshold": very_similar_rms_threshold
        }

    def initialize(self, initial_pool: list[dict]):
        # 将初始池评估并分发给 C 和 B
        evaluated_pool = []
        for ind in initial_pool:
            ind = evaluate_constraints(ind, self.constraints)
            evaluated_pool.append(ind)

        self.pop_c = [ind for ind in evaluated_pool if ind["is_valid"]]
        self.pop_b = [ind for ind in evaluated_pool if not ind["is_valid"]]
        # 初始 A 种群直接复制全局
        self.pop_a = copy.deepcopy(evaluated_pool)

        # 初始截断
        self._update_pop_c([])
        self._update_pop_b([])
        self._update_pop_a([])

    def ask(self, n_offspring, parent_selection_mode, tournament_k, noise_level, model_path, limit_density) -> tuple[
        list, list]:
        """
        按照策略生成子代：
        - 25%: C 变异
        - 25%: B 变异
        - 25%: A 与 C 交叉
        - 25%: A 与 A 交叉
        """
        pending_denoise = []
        offspring_meta = []

        quota = max(1, n_offspring // 4)

        # 辅助取人函数
        def get_rand(pop, fallback):
            return random.choice(pop) if pop else random.choice(fallback)

        all_fallback = self.pop_c + self.pop_b + self.pop_a

        while len(offspring_meta) < n_offspring:
            # 1. C 变异
            if len(offspring_meta) < quota * 1:
                p = get_rand(self.pop_c, all_fallback)
                noisy = add_noise(p, t=noise_level, model_path_or_name=model_path)
                noisy["name"] = f"mut_C_{len(offspring_meta):03d}_noisy.cif"
                pending_denoise.append(noisy)
                offspring_meta.append({"type": "mutation_C", "parent_1": p.get("name"), "parent_2": ""})
                continue

            # 2. B 变异
            if len(offspring_meta) < quota * 2:
                p = get_rand(self.pop_b, all_fallback)
                noisy = add_noise(p, t=noise_level, model_path_or_name=model_path)
                noisy["name"] = f"mut_B_{len(offspring_meta):03d}_noisy.cif"
                pending_denoise.append(noisy)
                offspring_meta.append({"type": "mutation_B", "parent_1": p.get("name"), "parent_2": ""})
                continue

            # 3. A 与 C 交叉
            if len(offspring_meta) < quota * 3:
                pa = get_rand(self.pop_a, all_fallback)
                pc = get_rand(self.pop_c, all_fallback)
                noisy_a = add_noise(pa, t=noise_level, model_path_or_name=model_path)
                noisy_c = add_noise(pc, t=noise_level, model_path_or_name=model_path)
                cross = crossover(noisy_a, noisy_c, limit_density=limit_density)
                cross["name"] = f"cross_AC_{len(offspring_meta):03d}_noisy.cif"
                pending_denoise.append(cross)
                offspring_meta.append({"type": "crossover_AC", "parent_1": pa.get("name"), "parent_2": pc.get("name")})
                continue

            # 4. A 与 A 交叉
            pa1 = get_rand(self.pop_a, all_fallback)
            pa2 = get_rand(self.pop_a, all_fallback)
            noisy_a1 = add_noise(pa1, t=noise_level, model_path_or_name=model_path)
            noisy_a2 = add_noise(pa2, t=noise_level, model_path_or_name=model_path)
            cross = crossover(noisy_a1, noisy_a2, limit_density=limit_density)
            cross["name"] = f"cross_AA_{len(offspring_meta):03d}_noisy.cif"
            pending_denoise.append(cross)
            offspring_meta.append({"type": "crossover_AA", "parent_1": pa1.get("name"), "parent_2": pa2.get("name")})

        return pending_denoise, offspring_meta

    def tell(self, evaluated_offspring: list[dict]):
        # 所有子代经过约束判定后分流
        for ind in evaluated_offspring:
            evaluate_constraints(ind, self.constraints)

        new_feasible = [ind for ind in evaluated_offspring if ind["is_valid"]]
        new_infeasible = [ind for ind in evaluated_offspring if not ind["is_valid"]]

        # 三个种群分别进行自己独特的环境更新！
        self._update_pop_c(new_feasible)
        self._update_pop_b(new_infeasible)
        self._update_pop_a(evaluated_offspring)  # A 照单全收

    def _update_pop_c(self, new_inds):
        candidates = self.pop_c + new_inds
        if not candidates: return
        # 剪枝 + SPEA2
        candidates = prune_similar_population(candidates, **self.prune_params)
        self.pop_c = spea2_selection(candidates, self.objectives, self.max_size_c)

    def _update_pop_b(self, new_inds):
        candidates = self.pop_b + new_inds
        if not candidates: return
        # CDP-FNDS 优化不可行解，逼迫走向可行域
        fast_non_dominated_sorting(candidates, self.objectives)
        candidates = prune_similar_population(candidates, **self.prune_params)
        self.pop_b = candidates[:self.max_size_b]

    def _update_pop_a(self, new_inds):
        """
        Pop A 顶级多样性策略：
        1. 计算每个解被 Pop C 支配的次数 (c_dom_count)。
        2. 按 c_dom_count 分组，组内计算目标空间的拥挤距离。
        3. 综合评分：绝对优先被 C 支配最少的解；同级别下优先目标空间最孤立的解。
        4. 结构空间剪枝：保证同等优秀的解中，晶体结构彼此远离。
        """
        candidates = self.pop_a + new_inds
        if not candidates: return

        # 1. 计算每个候选解被 Pop C 中的多少个精英支配
        for cand in candidates:
            dom_count = 0
            if self.pop_c:
                for c in self.pop_c:
                    if dominates_cdp(c, cand, self.objectives):
                        dom_count += 1
            cand["c_domination_count"] = dom_count

        # 2. 按被支配次数分组，在各组“内部”计算目标空间拥挤距离
        # 这样可以保证，即便是劣势组，内部也有自己的孤立点探索优势
        from collections import defaultdict
        groups = defaultdict(list)
        for cand in candidates:
            groups[cand["c_domination_count"]].append(cand)

        for count, group in groups.items():
            calculate_objective_crowding_distance(group, self.objectives)

        # 3. 计算综合适应度 (降维打击)
        # 惩罚项：被 C 支配的次数 * 10亿 (绝对优先级)
        # 奖励项：加上自身在组内的拥挤距离 (次级优先级)
        for cand in candidates:
            penalty = cand["c_domination_count"] * 1e9
            bonus = cand.get("crowding_distance", 0.0)
            cand["fitness"] = -penalty + bonus

        # 4. 排序并执行结构空间剪枝
        # 由于我们设计的 fitness 完美包含了“不被C压制”和“目标空间发散”，
        # 接下来 prune_similar_population 只需要从头往下挑，遇到结构长得太像的直接砍！
        sorted_cands = sorted(candidates, key=lambda x: x["fitness"], reverse=True)
        pruned_cands = prune_similar_population(sorted_cands, **self.prune_params)

        self.pop_a = pruned_cands[:self.max_size_a]

    def get_current_population(self) -> list[dict]:
        for ind in self.pop_c: ind["pop_type"] = "Pop_C (Feasible_Elite)"
        for ind in self.pop_b: ind["pop_type"] = "Pop_B (Infeasible_Guide)"
        for ind in self.pop_a: ind["pop_type"] = "Pop_A (Diverse_Explore)"
        return self.pop_c + self.pop_b + self.pop_a