"""Structural, inheritance, diversity, and protocol diagnostics.

The functions in this module are deliberately independent from the evolutionary
operators.  They observe populations and parent/offspring relationships without
changing molecule generation, property prediction, or constraint evaluation.
"""

from __future__ import annotations

from functools import lru_cache
import itertools
import math
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence


def _valid_tensors(molecule: Mapping[str, Any]):
    import torch

    positions = molecule["x"]
    atom_types = molecule["atom_type"]
    mask = molecule.get("node_mask")
    if mask is not None:
        valid = mask.squeeze(-1).bool()
        positions = positions[valid]
        atom_types = atom_types[valid]
    elif "long" in molecule:
        length = int(molecule["long"])
        positions = positions[:length]
        atom_types = atom_types[:length]
    positions = positions.detach().cpu().to(torch.float64)
    atom_types = atom_types.detach().cpu().to(torch.long)
    return positions, atom_types


def molecule_to_rdkit(
    molecule: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
    *,
    sanitize: bool = True,
):
    """Convert one generated tensor molecule to an RDKit molecule with 3D coordinates."""

    from rdkit import Chem
    from qm9.rdkit_functions import bond_dict, build_xae_molecule

    positions, atom_types = _valid_tensors(molecule)
    if positions.shape[0] == 0:
        return None
    _, adjacency, edge_types = build_xae_molecule(
        positions,
        atom_types,
        dataset_info,
    )
    decoder = dataset_info["atom_decoder"]
    editable = Chem.RWMol()
    for atom_type in atom_types:
        editable.AddAtom(Chem.Atom(str(decoder[int(atom_type.item())])))
    for row, col in adjacency.nonzero():
        i = int(row.item())
        j = int(col.item())
        if i <= j:
            continue
        order = int(edge_types[i, j].item())
        if order:
            editable.AddBond(i, j, bond_dict[order])
    result = editable.GetMol()
    conformer = Chem.Conformer(result.GetNumAtoms())
    for index, xyz in enumerate(positions.tolist()):
        conformer.SetAtomPosition(index, tuple(float(value) for value in xyz))
    result.AddConformer(conformer, assignId=True)
    if sanitize:
        try:
            Chem.SanitizeMol(result)
        except Exception:
            return None
    return result


def _heavy_molecule(molecule):
    from rdkit import Chem

    if molecule is None:
        return None
    try:
        return Chem.RemoveHs(molecule, sanitize=True)
    except Exception:
        return molecule


def canonical_smiles(molecule) -> str | None:
    from rdkit import Chem

    heavy = _heavy_molecule(molecule)
    if heavy is None:
        return None
    try:
        return Chem.MolToSmiles(heavy, canonical=True)
    except Exception:
        return None


def morgan_fingerprint(molecule):
    from rdkit.Chem import rdFingerprintGenerator

    heavy = _heavy_molecule(molecule)
    if heavy is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    try:
        return generator.GetFingerprint(heavy)
    except Exception:
        return None


def murcko_smiles(molecule) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    heavy = _heavy_molecule(molecule)
    if heavy is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(heavy)
        if scaffold.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def _brics_fragments(molecule) -> list[Any]:
    from rdkit import Chem
    from rdkit.Chem import BRICS

    heavy = _heavy_molecule(molecule)
    if heavy is None:
        return []
    try:
        bond_indices = [
            heavy.GetBondBetweenAtoms(int(atoms[0]), int(atoms[1])).GetIdx()
            for atoms, _ in BRICS.FindBRICSBonds(heavy)
        ]
        if not bond_indices:
            return []
        fragmented = Chem.FragmentOnBonds(
            heavy,
            bond_indices,
            addDummies=False,
        )
        return [
            fragment
            for fragment in Chem.GetMolFrags(
                fragmented,
                asMols=True,
                sanitizeFrags=True,
            )
            if fragment.GetNumHeavyAtoms() >= 2
        ]
    except Exception:
        return []


def _mcs(parent, child, timeout: int = 2):
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    parent = _heavy_molecule(parent)
    child = _heavy_molecule(child)
    if parent is None or child is None:
        return None, None, None
    try:
        result = rdFMCS.FindMCS(
            [parent, child],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=timeout,
        )
        if result.canceled or result.numAtoms == 0:
            return result, None, None
        query = Chem.MolFromSmarts(result.smartsString)
        return result, parent.GetSubstructMatch(query), child.GetSubstructMatch(query)
    except Exception:
        return None, None, None


def _kabsch_rmsd(parent, child, parent_match, child_match) -> float | None:
    import numpy as np

    if not parent_match or not child_match or len(parent_match) != len(child_match):
        return None
    try:
        parent_conf = parent.GetConformer()
        child_conf = child.GetConformer()
        parent_xyz = np.asarray(
            [
                list(parent_conf.GetAtomPosition(int(index)))
                for index in parent_match
            ],
            dtype=float,
        )
        child_xyz = np.asarray(
            [
                list(child_conf.GetAtomPosition(int(index)))
                for index in child_match
            ],
            dtype=float,
        )
        parent_xyz -= parent_xyz.mean(axis=0, keepdims=True)
        child_xyz -= child_xyz.mean(axis=0, keepdims=True)
        covariance = child_xyz.T @ parent_xyz
        left, _, right = np.linalg.svd(covariance)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        aligned = child_xyz @ rotation
        return float(np.sqrt(np.mean(np.sum((aligned - parent_xyz) ** 2, axis=1))))
    except Exception:
        return None


def _objective_utility(molecule: Mapping[str, Any], spec: Any) -> float | None:
    values = []
    for objective in spec.objectives:
        key = f"__canonical__{objective.name}"
        if key not in molecule:
            return None
        lower, upper = objective.bounds
        raw = float(molecule[key])
        if objective.direction == "max":
            utility = (raw - lower) / (upper - lower)
        else:
            utility = (upper - raw) / (upper - lower)
        values.append(max(0.0, min(1.0, utility)))
    return sum(values) / len(values) if values else None


def inheritance_metrics(
    child: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]],
    dataset_info: Mapping[str, Any],
    spec: Any,
) -> dict[str, float | None]:
    """Measure parent-to-offspring inheritance without affecting selection."""

    from rdkit import Chem, DataStructs

    child_mol = molecule_to_rdkit(child, dataset_info)
    child_fp = morgan_fingerprint(child_mol)
    child_utility = _objective_utility(child, spec)
    per_parent: list[dict[str, float | None]] = []
    for parent in parents:
        parent_mol = molecule_to_rdkit(parent, dataset_info)
        parent_fp = morgan_fingerprint(parent_mol)
        morgan = (
            float(DataStructs.TanimotoSimilarity(parent_fp, child_fp))
            if parent_fp is not None and child_fp is not None
            else 0.0
        )
        mcs_result, parent_match, child_match = _mcs(parent_mol, child_mol)
        parent_atoms = (
            max(1, _heavy_molecule(parent_mol).GetNumAtoms())
            if parent_mol is not None
            else 1
        )
        mcs_fraction = (
            float(mcs_result.numAtoms / parent_atoms)
            if mcs_result is not None
            else 0.0
        )
        scaffold = murcko_smiles(parent_mol)
        scaffold_retention = 0.0
        if scaffold and child_mol is not None:
            query = Chem.MolFromSmiles(scaffold)
            scaffold_retention = float(
                bool(query is not None and _heavy_molecule(child_mol).HasSubstructMatch(query))
            )
        elif parent_mol is not None and child_mol is not None:
            # Acyclic parents have an empty Murcko scaffold.  Retention then
            # means that the child remains acyclic, rather than dropping the
            # metric from the JSONL record.
            scaffold_retention = float(murcko_smiles(child_mol) is None)
        fragments = _brics_fragments(parent_mol)
        fragment_retention = 0.0
        if fragments and child_mol is not None:
            child_heavy = _heavy_molecule(child_mol)
            fragment_retention = sum(
                child_heavy.HasSubstructMatch(fragment)
                for fragment in fragments
            ) / len(fragments)
        elif parent_mol is not None and child_mol is not None:
            # With no BRICS-cleavable bond, use retention of the intact parent
            # heavy-atom graph as the single-fragment convention.
            fragment_retention = float(
                _heavy_molecule(child_mol).HasSubstructMatch(
                    _heavy_molecule(parent_mol)
                )
            )
        rmsd = _kabsch_rmsd(
            _heavy_molecule(parent_mol),
            _heavy_molecule(child_mol),
            parent_match,
            child_match,
        )
        parent_utility = _objective_utility(parent, spec)
        improvement = (
            child_utility - parent_utility
            if child_utility is not None and parent_utility is not None
            else None
        )
        per_parent.append(
            {
                "morgan_similarity": morgan,
                "mcs_atom_fraction": mcs_fraction,
                "murcko_retention": scaffold_retention,
                "fragment_retention": fragment_retention,
                "mcs_3d_rmsd": rmsd,
                "objective_improvement": improvement,
            }
        )

    result: dict[str, float | None] = {}
    names = (
        "morgan_similarity",
        "mcs_atom_fraction",
        "murcko_retention",
        "fragment_retention",
        "objective_improvement",
    )
    for name in names:
        values = [
            float(record[name])
            for record in per_parent
            if record[name] is not None
        ]
        result[name] = sum(values) / len(values) if values else None
        result[f"{name}_max"] = max(values) if values else None
    rmsds = [
        float(record["mcs_3d_rmsd"])
        for record in per_parent
        if record["mcs_3d_rmsd"] is not None
    ]
    result["mcs_3d_rmsd"] = sum(rmsds) / len(rmsds) if rmsds else None
    result["mcs_3d_rmsd_min"] = min(rmsds) if rmsds else None

    inheritance_components = [
        result.get("morgan_similarity"),
        result.get("mcs_atom_fraction"),
        result.get("murcko_retention"),
        result.get("fragment_retention"),
    ]
    if result.get("mcs_3d_rmsd") is not None:
        inheritance_components.append(
            math.exp(-float(result["mcs_3d_rmsd"]) / 2.0)
        )
    numeric = [
        float(value)
        for value in inheritance_components
        if value is not None
    ]
    result["inheritance_score"] = (
        sum(numeric) / len(numeric) if numeric else 0.0
    )
    return result


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "inheritance",
) -> dict[str, float]:
    result: dict[str, float] = {}
    keys = sorted(
        {
            str(key)
            for record in records
            for key, value in record.items()
            if isinstance(value, (int, float)) and value is not None
        }
    )
    for key in keys:
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
            and record.get(key) is not None
        ]
        if values:
            result[f"{prefix}.{key}.mean"] = sum(values) / len(values)
            result[f"{prefix}.{key}.min"] = min(values)
            result[f"{prefix}.{key}.max"] = max(values)
    result[f"{prefix}.n"] = float(len(records))
    return result


def population_diversity_metrics(
    population: Sequence[Mapping[str, Any]],
    dataset_info: Mapping[str, Any],
    *,
    max_mcs_pairs: int = 64,
) -> dict[str, float]:
    import numpy as np
    from rdkit import DataStructs

    molecules = [
        molecule_to_rdkit(item, dataset_info)
        for item in population
    ]
    valid = [molecule for molecule in molecules if molecule is not None]
    smiles = [value for value in (canonical_smiles(mol) for mol in valid) if value]
    scaffolds = [value for value in (murcko_smiles(mol) for mol in valid) if value]
    fingerprints = [fp for fp in (morgan_fingerprint(mol) for mol in valid) if fp]
    similarities = []
    nearest_distances = []
    for index, fingerprint in enumerate(fingerprints):
        row = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
        others = [
            float(value)
            for other_index, value in enumerate(row)
            if other_index != index
        ]
        if others:
            nearest_distances.append(1.0 - max(others))
        similarities.extend(
            float(row[other_index])
            for other_index in range(index + 1, len(row))
        )

    pair_indices = list(itertools.combinations(range(len(valid)), 2))
    if len(pair_indices) > max_mcs_pairs:
        selected = np.linspace(
            0,
            len(pair_indices) - 1,
            num=max_mcs_pairs,
            dtype=int,
        )
        pair_indices = [pair_indices[int(index)] for index in selected]
    redundant = 0
    measured = 0
    for left, right in pair_indices:
        mcs_result, _, _ = _mcs(valid[left], valid[right], timeout=1)
        if mcs_result is None:
            continue
        denominator = max(
            1,
            min(
                _heavy_molecule(valid[left]).GetNumAtoms(),
                _heavy_molecule(valid[right]).GetNumAtoms(),
            ),
        )
        measured += 1
        redundant += (mcs_result.numAtoms / denominator) >= 0.8

    nearest = np.asarray(nearest_distances, dtype=float)
    mean_similarity = (
        sum(similarities) / len(similarities) if similarities else 0.0
    )
    return {
        "valid_structure_rate": len(valid) / len(population) if population else 0.0,
        "unique_smiles": float(len(set(smiles))),
        "unique_smiles_rate": len(set(smiles)) / len(smiles) if smiles else 0.0,
        "scaffold_count": float(len(set(scaffolds))),
        "scaffold_diversity": len(set(scaffolds)) / len(scaffolds) if scaffolds else 0.0,
        "internal_tanimoto": mean_similarity,
        "internal_tanimoto_diversity": 1.0 - mean_similarity,
        "nearest_neighbor_distance_mean": (
            float(nearest.mean()) if nearest.size else 0.0
        ),
        "nearest_neighbor_distance_std": (
            float(nearest.std()) if nearest.size else 0.0
        ),
        "nearest_neighbor_distance_q10": (
            float(np.quantile(nearest, 0.1)) if nearest.size else 0.0
        ),
        "nearest_neighbor_distance_q50": (
            float(np.quantile(nearest, 0.5)) if nearest.size else 0.0
        ),
        "nearest_neighbor_distance_q90": (
            float(np.quantile(nearest, 0.9)) if nearest.size else 0.0
        ),
        "mcs_redundancy_rate": redundant / measured if measured else 0.0,
    }


def novelty_against_reference(
    candidates: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    dataset_info: Mapping[str, Any],
) -> dict[str, float]:
    """Measure exact and nearest-neighbor novelty against a prior population."""

    from rdkit import DataStructs

    reference_molecules = [
        molecule_to_rdkit(item, dataset_info)
        for item in reference
    ]
    reference_smiles = {
        value
        for value in (
            canonical_smiles(molecule)
            for molecule in reference_molecules
        )
        if value
    }
    reference_fingerprints = [
        fingerprint
        for fingerprint in (
            morgan_fingerprint(molecule)
            for molecule in reference_molecules
        )
        if fingerprint is not None
    ]
    exact_novel = []
    fingerprint_novelty = []
    for item in candidates:
        molecule = molecule_to_rdkit(item, dataset_info)
        smiles = canonical_smiles(molecule)
        if smiles:
            exact_novel.append(float(smiles not in reference_smiles))
        fingerprint = morgan_fingerprint(molecule)
        if fingerprint is not None and reference_fingerprints:
            similarity = max(
                float(value)
                for value in DataStructs.BulkTanimotoSimilarity(
                    fingerprint,
                    reference_fingerprints,
                )
            )
            fingerprint_novelty.append(1.0 - similarity)
    return {
        "offspring_exact_novelty_rate": (
            sum(exact_novel) / len(exact_novel) if exact_novel else 0.0
        ),
        "offspring_fingerprint_novelty_mean": (
            sum(fingerprint_novelty) / len(fingerprint_novelty)
            if fingerprint_novelty
            else 0.0
        ),
    }


@lru_cache(maxsize=2)
def _training_reference(path: str):
    from rdkit import Chem

    with Path(path).open("rb") as handle:
        smiles_values = pickle.load(handle)
    fingerprints = []
    scaffolds = set()
    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(str(smiles))
        fingerprint = morgan_fingerprint(molecule)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
        scaffold = murcko_smiles(molecule)
        if scaffold:
            scaffolds.add(scaffold)
    return tuple(fingerprints), frozenset(scaffolds)


def training_reference_metrics(
    population: Sequence[Mapping[str, Any]],
    dataset_info: Mapping[str, Any],
    reference_path: str | Path,
) -> dict[str, float]:
    import numpy as np
    from rdkit import DataStructs

    reference_fingerprints, reference_scaffolds = _training_reference(
        str(Path(reference_path).resolve())
    )
    nearest = []
    generated_scaffolds = []
    for item in population:
        molecule = molecule_to_rdkit(item, dataset_info)
        fingerprint = morgan_fingerprint(molecule)
        if fingerprint is not None and reference_fingerprints:
            nearest.append(
                max(
                    float(value)
                    for value in DataStructs.BulkTanimotoSimilarity(
                        fingerprint,
                        reference_fingerprints,
                    )
                )
            )
        scaffold = murcko_smiles(molecule)
        if scaffold:
            generated_scaffolds.append(scaffold)
    nearest_array = np.asarray(nearest, dtype=float)
    novel_scaffolds = [
        scaffold
        for scaffold in generated_scaffolds
        if scaffold not in reference_scaffolds
    ]
    return {
        "train_nearest_tanimoto_mean": (
            float(nearest_array.mean()) if nearest_array.size else 0.0
        ),
        "train_nearest_tanimoto_max": (
            float(nearest_array.max()) if nearest_array.size else 0.0
        ),
        "train_nearest_tanimoto_q50": (
            float(np.quantile(nearest_array, 0.5))
            if nearest_array.size
            else 0.0
        ),
        "training_novelty_mean": (
            float((1.0 - nearest_array).mean())
            if nearest_array.size
            else 0.0
        ),
        "scaffold_novelty": (
            len(novel_scaffolds) / len(generated_scaffolds)
            if generated_scaffolds
            else 0.0
        ),
    }


def _normalize_distance(matrix):
    import numpy as np

    matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=1.0)
    maximum = float(matrix.max()) if matrix.size else 0.0
    if maximum > 0:
        matrix = matrix / maximum
    np.fill_diagonal(matrix, 0.0)
    return matrix


def structural_distance_matrix(
    population: Sequence[Mapping[str, Any]],
    dataset_info: Mapping[str, Any],
    mode: str,
):
    import numpy as np
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors
    from scipy.spatial.distance import cdist

    normalized_mode = str(mode).lower()
    molecules = [molecule_to_rdkit(item, dataset_info) for item in population]
    if normalized_mode in {"morgan", "mixed"}:
        fingerprints = [morgan_fingerprint(molecule) for molecule in molecules]
        morgan = np.zeros((len(population), len(population)), dtype=float)
        for row, fingerprint in enumerate(fingerprints):
            if fingerprint is None:
                morgan[row, :] = 1.0
                morgan[row, row] = 0.0
                continue
            for col in range(row + 1, len(fingerprints)):
                other = fingerprints[col]
                distance = (
                    1.0 - float(DataStructs.TanimotoSimilarity(fingerprint, other))
                    if other is not None
                    else 1.0
                )
                morgan[row, col] = morgan[col, row] = distance
    else:
        morgan = None

    if normalized_mode in {"usrcat", "mixed"}:
        descriptors = []
        for molecule in molecules:
            try:
                descriptors.append(
                    np.asarray(rdMolDescriptors.GetUSRCAT(molecule), dtype=float)
                )
            except Exception:
                descriptors.append(np.zeros(60, dtype=float))
        usrcat = _normalize_distance(cdist(descriptors, descriptors))
    else:
        usrcat = None

    if normalized_mode == "morgan":
        return morgan
    if normalized_mode == "usrcat":
        return usrcat
    if normalized_mode == "mixed":
        return 0.5 * morgan + 0.5 * usrcat
    if normalized_mode == "raw_coords":
        coordinate_rows = []
        max_atoms = max(
            (_valid_tensors(item)[0].shape[0] for item in population),
            default=0,
        )
        for item in population:
            positions, _ = _valid_tensors(item)
            array = positions.numpy()
            if len(array):
                array = array - array.mean(axis=0, keepdims=True)
            padded = np.zeros((max_atoms, 3), dtype=float)
            padded[: len(array)] = array
            coordinate_rows.append(padded.reshape(-1))
        return _normalize_distance(cdist(coordinate_rows, coordinate_rows))
    raise ValueError(f"Unsupported structural distance mode: {mode}")


def _constraint_dominance_rank(
    population: Sequence[Mapping[str, Any]],
    objective_names: Sequence[str],
):
    import numpy as np

    objectives = np.asarray(
        [
            [
                float(item[name].item())
                if hasattr(item[name], "item")
                else float(item[name])
                for name in objective_names
            ]
            for item in population
        ],
        dtype=float,
    )
    violations = []
    for item in population:
        chemical = max(0.0, float(item.get("Constraint", 0.0)))
        structural = item.get("structure_Con", 0.0)
        if isinstance(structural, (list, tuple)):
            structural = sum(float(value) for value in structural)
        violations.append(chemical + max(0.0, float(structural)))
    violations = np.asarray(violations, dtype=float)
    dominates = np.zeros((len(population), len(population)), dtype=bool)
    for left in range(len(population)):
        for right in range(len(population)):
            if left == right:
                continue
            if violations[left] < violations[right]:
                dominates[left, right] = True
            elif violations[left] == violations[right]:
                dominates[left, right] = bool(
                    np.any(objectives[left] < objectives[right])
                    and np.all(objectives[left] <= objectives[right])
                )
    strengths = dominates.sum(axis=1)
    return np.asarray(
        [
            strengths[dominates[:, index]].sum()
            for index in range(len(population))
        ],
        dtype=float,
    )


def select_population(
    population: list[dict[str, Any]],
    size: int,
    objective_names: Sequence[str],
    dataset_info: Mapping[str, Any],
    mode: str,
    min_distance: float,
) -> list[dict[str, Any]]:
    """SAES-compatible selection with an explicit distance ablation mode."""

    import numpy as np

    if len(population) <= size:
        return population
    normalized_mode = str(mode).lower()
    if normalized_mode == "objective":
        import evo

        return evo.EnvironmentalSelectionCon(
            population,
            size,
            list(objective_names),
        )
    distances = structural_distance_matrix(
        population,
        dataset_info,
        normalized_mode,
    )
    ranks = _constraint_dominance_rank(population, objective_names)
    selected: list[int] = []
    unselected = list(range(len(population)))
    minimum_rank = float(ranks.min())
    candidates = [index for index in unselected if ranks[index] == minimum_rank]
    first = (
        candidates[int(np.argmax(np.mean(distances[candidates, :], axis=1)))]
        if len(candidates) > 1
        else candidates[0]
    )
    selected.append(first)
    unselected.remove(first)
    while len(selected) < size and unselected:
        distance_to_selected = np.min(
            distances[unselected][:, selected],
            axis=1,
        )
        valid = distance_to_selected >= float(min_distance)
        if np.any(valid):
            valid_indices = np.asarray(unselected)[valid]
            valid_ranks = ranks[valid_indices]
            best_rank = float(valid_ranks.min())
            tied = valid_indices[valid_ranks == best_rank]
            tied_distances = distance_to_selected[valid][valid_ranks == best_rank]
            target = int(tied[int(np.argmax(tied_distances))])
        else:
            best_rank = float(ranks[unselected].min())
            tied = [
                index
                for index in unselected
                if ranks[index] == best_rank
            ]
            if len(tied) == 1:
                target = tied[0]
            else:
                tied_distances = np.min(
                    distances[tied][:, selected],
                    axis=1,
                )
                target = tied[int(np.argmax(tied_distances))]
        selected.append(target)
        unselected.remove(target)
    for position, index in enumerate(selected):
        population[index]["fitness"] = position / max(1, size)
    for index in unselected:
        population[index]["fitness"] = 1.0 + float(ranks[index])
    return [population[index] for index in selected]


def make_inheritance_scheduler(evo_module: Any, **kwargs: Any):
    """Create the inheritance-aware variant without modifying the core scheduler."""

    class InheritanceAwareNoiseScheduler(evo_module.AdaptiveNoiseScheduler):
        def _calculate_score(self, population):
            integrity = super()._calculate_score(population)
            values = [
                float(item.get("_inheritance_score", 0.0))
                for item in population
                if item.get("_inheritance_score") is not None
            ]
            inheritance = sum(values) / len(values) if values else 1.0
            return math.sqrt(max(0.0, integrity) * max(0.0, inheritance))

    return InheritanceAwareNoiseScheduler(**kwargs)
