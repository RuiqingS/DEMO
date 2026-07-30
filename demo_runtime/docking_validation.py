"""Pose, force-field, docking, and interaction checks for the bundled CD set.

This module uses PoseBusters, MMFF/UFF, Open Babel, and QVina.  It deliberately
does not invoke xTB or DFT.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from .recording import JsonlRunRecorder, canonical_hash, get_env_recorder


REPO_ROOT = Path(__file__).resolve().parents[1]
_VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
        "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98}
_POSITIVE_RESIDUES = {"ARG", "LYS", "HIS"}
_NEGATIVE_RESIDUES = {"ASP", "GLU"}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_ligand(path: Path):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    return next((molecule for molecule in supplier if molecule is not None), None)


def _coordinates(molecule, *, heavy_only: bool = True):
    import numpy as np

    conformer = molecule.GetConformer()
    indices = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if not heavy_only or atom.GetAtomicNum() > 1
    ]
    return np.asarray(
        [
            list(conformer.GetAtomPosition(index))
            for index in indices
        ],
        dtype=float,
    )


def _aligned_rmsd(reference, candidate) -> float:
    import numpy as np

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    count = min(len(reference), len(candidate))
    if count == 0:
        return 0.0
    reference = reference[:count] - reference[:count].mean(axis=0)
    candidate = candidate[:count] - candidate[:count].mean(axis=0)
    left, _, right = np.linalg.svd(candidate.T @ reference)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    delta = candidate @ rotation - reference
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def relax_ligand(molecule):
    """Return relaxed molecule, force-field name, energies, and heavy-atom RMSD."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    relaxed = Chem.Mol(molecule)
    reference = _coordinates(relaxed)
    relaxed = Chem.AddHs(relaxed, addCoords=True)
    properties = None
    force_field = None
    if AllChem.MMFFHasAllMoleculeParams(relaxed):
        properties = AllChem.MMFFGetMoleculeProperties(relaxed)
        force_field = AllChem.MMFFGetMoleculeForceField(relaxed, properties)
        force_field_name = "MMFF94"
    else:
        force_field = AllChem.UFFGetMoleculeForceField(relaxed)
        force_field_name = "UFF"
    if force_field is None:
        raise ValueError("RDKit could not construct an MMFF or UFF force field")
    energy_before = float(force_field.CalcEnergy())
    force_field.Minimize(maxIts=500)
    energy_after = float(force_field.CalcEnergy())
    heavy_relaxed = Chem.RemoveHs(relaxed)
    rmsd = _aligned_rmsd(reference, _coordinates(heavy_relaxed))
    return (
        heavy_relaxed,
        force_field_name,
        energy_before,
        energy_after,
        rmsd,
    )


def _write_sdf(molecule, path: Path) -> None:
    from rdkit import Chem

    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()


def _protein_atoms(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            element = line[76:78].strip().upper()
            if not element:
                element = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()[:1]
            atoms.append(
                {
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "element": element,
                    "residue": line[17:20].strip().upper(),
                }
            )
        except ValueError:
            continue
    return atoms


def _ligand_atoms(molecule) -> list[dict[str, Any]]:
    conformer = molecule.GetConformer()
    result = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        result.append(
            {
                "x": float(position.x),
                "y": float(position.y),
                "z": float(position.z),
                "element": atom.GetSymbol().upper(),
                "formal_charge": int(atom.GetFormalCharge()),
            }
        )
    return result


def interaction_metrics(
    protein_atoms: Sequence[Mapping[str, Any]],
    ligand_atoms: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute an independent, geometry-only interaction fingerprint and score."""

    contacts = clashes = hbonds = hydrophobic = ionic = 0
    for ligand in ligand_atoms:
        ligand_element = str(ligand.get("element", "")).upper()
        ligand_charge = int(ligand.get("formal_charge", 0) or 0)
        for protein in protein_atoms:
            protein_element = str(protein.get("element", "")).upper()
            distance = math.sqrt(
                (float(ligand["x"]) - float(protein["x"])) ** 2
                + (float(ligand["y"]) - float(protein["y"])) ** 2
                + (float(ligand["z"]) - float(protein["z"])) ** 2
            )
            radius = _VDW.get(ligand_element, 1.7) + _VDW.get(
                protein_element, 1.7
            )
            clashes += distance < 0.75 * radius
            contacts += 2.0 <= distance <= 4.5
            hbonds += (
                ligand_element in {"N", "O", "S"}
                and protein_element in {"N", "O", "S"}
                and 2.0 <= distance <= 3.5
            )
            hydrophobic += (
                ligand_element in {"C", "S"}
                and protein_element in {"C", "S"}
                and 2.5 <= distance <= 4.5
            )
            residue = str(protein.get("residue", "")).upper()
            ionic += (
                distance <= 4.0
                and (
                    (ligand_charge < 0 and residue in _POSITIVE_RESIDUES)
                    or (ligand_charge > 0 and residue in _NEGATIVE_RESIDUES)
                )
            )
    score = (
        2.0 * clashes
        - 0.10 * contacts
        - 0.50 * hbonds
        - 0.05 * hydrophobic
        - 1.00 * ionic
    )
    return {
        "contact_count": float(contacts),
        "clash_count": float(clashes),
        "hbond_count": float(hbonds),
        "hydrophobic_count": float(hydrophobic),
        "ionic_count": float(ionic),
        "interaction_rescore": float(score),
    }


def _pdbqt_models(path: Path) -> list[list[dict[str, Any]]]:
    models: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith(("ATOM  ", "HETATM")):
            try:
                element = line[76:78].strip().upper()
                if not element:
                    element = line.split()[-1].upper()
                current.append(
                    {
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "element": element,
                        "formal_charge": 0,
                    }
                )
            except (ValueError, IndexError):
                continue
        elif line.startswith("ENDMDL") and current:
            models.append(current)
            current = []
    if current:
        models.append(current)
    return models


def binding_mode_diversity(models: Sequence[Sequence[Mapping[str, Any]]]) -> float:
    import numpy as np

    distances = []
    for left in range(len(models)):
        for right in range(left + 1, len(models)):
            left_xyz = np.asarray(
                [[atom["x"], atom["y"], atom["z"]] for atom in models[left]]
            )
            right_xyz = np.asarray(
                [[atom["x"], atom["y"], atom["z"]] for atom in models[right]]
            )
            distances.append(_aligned_rmsd(left_xyz, right_xyz))
    return sum(distances) / len(distances) if distances else 0.0


def _posebusters(
    ligand_path: Path,
    reference_path: Path,
    pocket_path: Path,
) -> dict[str, Any]:
    from posebusters import PoseBusters

    table = PoseBusters(config="dock").bust(
        mol_pred=str(ligand_path),
        mol_true=str(reference_path),
        mol_cond=str(pocket_path),
        full_report=True,
    )
    if len(table) == 0:
        return {"checks": 0, "passed": 0, "pass_rate": 0.0}
    row = table.iloc[0]
    boolean_values = [
        bool(value)
        for value in row.tolist()
        if isinstance(value, (bool,)) or type(value).__name__ == "bool_"
    ]
    return {
        "checks": len(boolean_values),
        "passed": sum(boolean_values),
        "pass_rate": (
            sum(boolean_values) / len(boolean_values)
            if boolean_values
            else 0.0
        ),
    }


def _convert_to_pdbqt(source: Path, target: Path, executable: str) -> None:
    subprocess.run(
        [executable, str(source), "-O", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )


def _run_qvina(
    receptor: Path,
    ligand: Path,
    output: Path,
    center: Sequence[float],
    executable: str,
    exhaustiveness: int,
    box_size: float,
) -> tuple[list[float], str]:
    command = [
        executable,
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--center_x",
        str(float(center[0])),
        "--center_y",
        str(float(center[1])),
        "--center_z",
        str(float(center[2])),
        "--size_x",
        str(box_size),
        "--size_y",
        str(box_size),
        "--size_z",
        str(box_size),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        "9",
        "--out",
        str(output),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    scores = []
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*\d+\s+(-?\d+(?:\.\d+)?)\s+", line)
        if match:
            scores.append(float(match.group(1)))
    return scores, result.stdout


def _discover(
    complex_root: Path,
    receptor_root: Path,
) -> list[tuple[str, Path, Path, Path]]:
    result = []
    for pocket in sorted(complex_root.glob("*-pocket10.pdb")):
        ligand_candidates = sorted(complex_root.glob(f"{pocket.stem}_*.sdf"))
        receptor = receptor_root / f"{pocket.stem}.pdbqt"
        if ligand_candidates and receptor.exists():
            result.append((pocket.stem, pocket, ligand_candidates[0], receptor))
    return result


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [
        float(record[key])
        for record in records
        if isinstance(record.get(key), (int, float))
    ]
    return sum(values) / len(values) if values else 0.0


def _pearson(records: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    import numpy as np

    pairs = [
        (float(record[left]), float(record[right]))
        for record in records
        if isinstance(record.get(left), (int, float))
        and isinstance(record.get(right), (int, float))
    ]
    if len(pairs) < 2:
        return 0.0
    array = np.asarray(pairs, dtype=float)
    if float(array[:, 0].std()) == 0.0 or float(array[:, 1].std()) == 0.0:
        return 0.0
    return float(np.corrcoef(array[:, 0], array[:, 1])[0, 1])


def run_validation(args: argparse.Namespace) -> JsonlRunRecorder:
    complex_root = _resolve(args.complex_root)
    receptor_root = _resolve(args.receptor_pdbqt_root)
    qvina = args.qvina or shutil.which("qvina2.1") or shutil.which("qvina2")
    obabel = args.obabel or shutil.which("obabel")
    if not qvina:
        raise FileNotFoundError("QVina executable qvina2.1 was not found")
    if not obabel:
        raise FileNotFoundError("Open Babel executable obabel was not found")

    recorder = get_env_recorder()
    if recorder is None:
        run_id = canonical_hash(
            {
                "complex_root": str(complex_root.resolve()),
                "receptor_root": str(receptor_root.resolve()),
                "seed": args.seed,
                "max_complexes": args.max_complexes,
            },
            length=24,
        )
        output = (
            _resolve(args.output)
            if args.output
            else REPO_ROOT / "results" / "docking_validation" / run_id
        )
        recorder = JsonlRunRecorder(
            output,
            run_id,
            {
                "suite": "reviewer_docking_validation",
                "problem": "bundled_cd_pose_validation",
                "method": "QVina+PoseBusters+MMFF/UFF",
                "seed": args.seed,
            },
        )
        recorder.event("run_start")

    artifacts = recorder.run_dir / "validation_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    complexes = _discover(complex_root, receptor_root)[: args.max_complexes]
    if not complexes:
        raise FileNotFoundError(
            f"No pocket/SDF/receptor triples found below {complex_root}"
        )

    records = []
    started = time.monotonic()
    for index, (complex_id, pocket, ligand_path, receptor) in enumerate(complexes):
        record: dict[str, Any] = {
            "record_type": "docking_validation",
            "complex_id": complex_id,
            "status": "failed",
        }
        try:
            ligand = _read_ligand(ligand_path)
            if ligand is None:
                raise ValueError(f"Cannot read ligand {ligand_path}")
            original_busters = _posebusters(ligand_path, ligand_path, pocket)
            relaxed, force_field, before, after, relaxation_rmsd = relax_ligand(
                ligand
            )
            relaxed_sdf = artifacts / f"{complex_id}.relaxed.sdf"
            relaxed_pdbqt = artifacts / f"{complex_id}.relaxed.pdbqt"
            docked_pdbqt = artifacts / f"{complex_id}.qvina.pdbqt"
            _write_sdf(relaxed, relaxed_sdf)
            relaxed_busters = _posebusters(relaxed_sdf, ligand_path, pocket)
            _convert_to_pdbqt(relaxed_sdf, relaxed_pdbqt, obabel)
            center = _coordinates(relaxed).mean(axis=0)
            scores, _ = _run_qvina(
                receptor,
                relaxed_pdbqt,
                docked_pdbqt,
                center,
                qvina,
                args.exhaustiveness,
                args.box_size,
            )
            models = _pdbqt_models(docked_pdbqt)
            protein = _protein_atoms(pocket)
            original_interactions = interaction_metrics(
                protein,
                _ligand_atoms(relaxed),
            )
            docked_interactions = (
                interaction_metrics(protein, models[0])
                if models
                else original_interactions
            )
            fingerprint_categories = (
                "hbond_count",
                "hydrophobic_count",
                "ionic_count",
            )
            active_categories = [
                name
                for name in fingerprint_categories
                if float(original_interactions[name]) > 0.0
            ]
            retained_categories = [
                name
                for name in active_categories
                if float(docked_interactions[name]) > 0.0
            ]
            category_retention = (
                len(retained_categories) / len(active_categories)
                if active_categories
                else 1.0
            )
            fingerprint_pass = float(
                category_retention >= 0.5
                and float(docked_interactions["clash_count"]) == 0.0
            )
            record.update(
                {
                    "status": "completed",
                    "force_field": force_field,
                    "force_field_energy_before": before,
                    "force_field_energy_after": after,
                    "strain_energy_release": before - after,
                    "relaxation_rmsd": relaxation_rmsd,
                    "posebusters_original_checks": original_busters["checks"],
                    "posebusters_original_pass_rate": original_busters[
                        "pass_rate"
                    ],
                    "posebusters_relaxed_checks": relaxed_busters["checks"],
                    "posebusters_relaxed_pass_rate": relaxed_busters[
                        "pass_rate"
                    ],
                    "qvina_best_score": min(scores) if scores else None,
                    "qvina_mode_count": len(models),
                    "binding_mode_diversity": binding_mode_diversity(models),
                    "interaction_fingerprint_category_retention": (
                        category_retention
                    ),
                    "interaction_fingerprint_pass": fingerprint_pass,
                    **{
                        f"original_{key}": value
                        for key, value in original_interactions.items()
                    },
                    **{
                        f"docked_{key}": value
                        for key, value in docked_interactions.items()
                    },
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        recorder.append("docking_validation", record)
        recorder.record_generation(
            [],
            [],
            {
                key: value
                for key, value in record.items()
                if isinstance(value, (int, float, bool))
            },
            generation=index,
            problem_id="bundled_cd_pose_validation_per_complex",
        )
        records.append(record)
        recorder.event(
            "docking_complex_end",
            complex_id=complex_id,
            status=record["status"],
            index=index,
        )

    completed = [record for record in records if record["status"] == "completed"]
    metrics = {
        "complex_count": len(records),
        "completed_complex_count": len(completed),
        "validation_success_rate": len(completed) / len(records),
        "posebusters_original_pass_rate": _mean(
            completed, "posebusters_original_pass_rate"
        ),
        "posebusters_relaxed_pass_rate": _mean(
            completed, "posebusters_relaxed_pass_rate"
        ),
        "relaxation_rmsd": _mean(completed, "relaxation_rmsd"),
        "strain_energy_release": _mean(completed, "strain_energy_release"),
        "qvina_best_score": _mean(completed, "qvina_best_score"),
        "binding_mode_diversity": _mean(
            completed, "binding_mode_diversity"
        ),
        "interaction_fingerprint_pass_rate": _mean(
            completed, "interaction_fingerprint_pass"
        ),
        "interaction_fingerprint_category_retention": _mean(
            completed, "interaction_fingerprint_category_retention"
        ),
        "rescoring_consistency_pearson": _pearson(
            completed, "qvina_best_score", "docked_interaction_rescore"
        ),
        "docked_hbond_count": _mean(completed, "docked_hbond_count"),
        "docked_hydrophobic_count": _mean(
            completed, "docked_hydrophobic_count"
        ),
        "docked_ionic_count": _mean(completed, "docked_ionic_count"),
        "docked_clash_count": _mean(completed, "docked_clash_count"),
        "wall_time_seconds": time.monotonic() - started,
        "validation_backend": "PoseBusters+MMFF/UFF+QVina+contact_rescore",
        "uses_xtb_or_dft": False,
    }
    recorder.record_generation(
        [],
        [],
        metrics,
        generation=len(records),
        problem_id="bundled_cd_pose_validation_aggregate",
    )
    recorder.event(
        "docking_validation_complete",
        status="completed",
        **metrics,
    )
    if get_env_recorder() is None:
        recorder.event("run_end", status="completed")
        recorder.mark_complete()
    return recorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complex-root", default="CDtest")
    parser.add_argument("--receptor-pdbqt-root", default="pdbqt/CD")
    parser.add_argument("--max-complexes", type=int, default=10)
    parser.add_argument("--exhaustiveness", type=int, default=2)
    parser.add_argument("--box-size", type=float, default=22.0)
    parser.add_argument("--qvina")
    parser.add_argument("--obabel")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DEMO_SEED", "42")))
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    run_validation(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
