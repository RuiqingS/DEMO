"""Audit imported legacy experiment definitions against their reference files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reference_compatibility.json"


def _normalized_text(
    path: Path,
    normalization: Mapping[str, str],
    ignore_unused_optimizer: bool = False,
) -> str:
    lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
    normalized: list[str] = []
    for line in lines:
        if normalization["backbone_import_pattern"] in line:
            continue
        backbone_override = re.match(
            r'^(?P<indent>\s*)use_EDM\s*=\s*env_bool\('
            r'"DEMO_USE_EDM",\s*default=(?P<default>True|False)\)$',
            line,
        )
        if backbone_override:
            line = (
                f"{backbone_override.group('indent')}use_EDM = "
                f"{backbone_override.group('default')}"
            )
        if (
            normalization["gpu_assignment_pattern"] in line
            or normalization["gpu_comment_pattern"] in line
        ):
            normalized.append("<GPU_VISIBILITY_MANAGED_BY_LAUNCHER>")
            continue
        if (
            ignore_unused_optimizer
            and normalization["unused_optimizer_pattern"] in line
        ):
            continue
        normalized.append(line.rstrip())
    return "\n".join(normalized).rstrip() + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_reference_compatibility(
    config_path: str | Path = DEFAULT_CONFIG,
    reference_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    normalization = config["normalization"]
    source_root = (
        Path(reference_root).resolve()
        if reference_root is not None
        else (REPO_ROOT / config["reference_root"]).resolve()
    )

    results: list[dict[str, Any]] = []
    for experiment in config["experiments"]:
        source = source_root / experiment["source"]
        target = REPO_ROOT / experiment["target"]
        record: dict[str, Any] = {
            "paper_role": experiment["paper_role"],
            "source": str(source),
            "target": str(target),
            "duplicate_command": bool(experiment.get("duplicate_command", False)),
        }
        if not source.exists():
            record["status"] = "missing_reference"
            results.append(record)
            continue
        if not target.exists():
            record["status"] = "missing_target"
            results.append(record)
            continue

        ignore_optimizer = bool(experiment.get("ignore_unused_optimizer", False))
        source_text = _normalized_text(
            source, normalization, ignore_unused_optimizer=ignore_optimizer
        )
        target_text = _normalized_text(
            target, normalization, ignore_unused_optimizer=ignore_optimizer
        )
        source_hash = _sha256(source_text)
        target_hash = _sha256(target_text)
        record.update(
            {
                "status": "match" if source_hash == target_hash else "mismatch",
                "source_sha256": source_hash,
                "target_sha256": target_hash,
            }
        )
        results.append(record)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference-root")
    parser.add_argument(
        "--allow-missing-reference",
        action="store_true",
        help="Do not fail when the external reference directory is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = audit_reference_compatibility(args.config, args.reference_root)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    failed = {
        "mismatch",
        "missing_target",
    }
    if not args.allow_missing_reference:
        failed.add("missing_reference")
    return 1 if any(record["status"] in failed for record in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
