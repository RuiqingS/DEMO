"""Build JSONL-backed paper tables from completed suite runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable, Mapping

from .recording import append_jsonl, read_jsonl


def _flatten(record: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in record.get("metrics", {}).items():
        if isinstance(value, (int, float)) and value is not None:
            result[str(name)] = float(value)
    for name, stats in record.get("objectives", {}).items():
        if not isinstance(stats, Mapping):
            continue
        for stat_name, value in stats.items():
            if isinstance(value, (int, float)) and value is not None:
                result[f"objective.{name}.{stat_name}"] = float(value)
    for name, stats in record.get("constraints", {}).items():
        if not isinstance(stats, Mapping):
            continue
        for stat_name, value in stats.items():
            if isinstance(value, (int, float)) and value is not None:
                result[f"constraint.{name}.{stat_name}"] = float(value)
    return result


def _final_records(suite_dir: Path) -> Iterable[dict[str, Any]]:
    for path in suite_dir.rglob("metrics.jsonl"):
        marker = path.parent / ".complete"
        if not marker.exists():
            continue
        run_id = marker.read_text(encoding="utf-8").strip()
        last_event = None
        for event in read_jsonl(path.parent / "events.jsonl"):
            last_event = event
        if not (
            run_id
            and last_event
            and last_event.get("run_id") == run_id
            and last_event.get("event") == "run_end"
            and last_event.get("status") == "completed"
        ):
            continue
        latest: dict[tuple[int, str, str], dict[str, Any]] = {}
        for record in read_jsonl(path):
            if record.get("record_type") != "generation":
                continue
            key = (
                int(record.get("internal_run_index", 0)),
                str(record.get("problem", "unknown")),
                str(record.get("population", "main")),
            )
            if key not in latest or int(record.get("generation", 0)) > int(
                latest[key].get("generation", -1)
            ):
                latest[key] = record
        yield from latest.values()


def _stats(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    avg = mean(values) if values else None
    std = stdev(values) if n > 1 else 0.0 if n == 1 else None
    radius = 1.96 * std / math.sqrt(n) if n > 1 and std is not None else 0.0
    return {
        "n": n,
        "mean": avg,
        "std": std,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "ci95_low": avg - radius if avg is not None else None,
        "ci95_high": avg + radius if avg is not None else None,
    }


def _format_cell(stats: Mapping[str, Any]) -> str:
    if not stats or stats.get("mean") is None:
        return "NA"
    return f"{stats['mean']:.4g} ± {stats['std']:.3g} (n={stats['n']})"


def summarize_suite(
    suite_dir: str | Path,
    suite_config: Mapping[str, Any] | None = None,
) -> Path:
    suite_path = Path(suite_dir)
    table_dir = suite_path / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in _final_records(suite_path):
        problem = str(record.get("problem", "unknown"))
        method = str(record.get("method", "unknown"))
        for metric, value in _flatten(record).items():
            values[(problem, method, metric)].append(value)

    summary_path = table_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()
    summarized: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (problem, method, metric), metric_values in sorted(values.items()):
        record = {
            "record_type": "summary",
            "problem": problem,
            "method": method,
            "metric": metric,
            **_stats(metric_values),
        }
        append_jsonl(summary_path, record)
        summarized[(problem, method, metric)] = record

    _write_long_csv(table_dir / "summary.csv", summarized.values())
    directions = _summary_directions(suite_config or {})
    metrics = sorted({metric for _, _, metric in summarized})
    for metric in metrics:
        _write_metric_tables(table_dir, metric, summarized, directions.get(metric, "max"))
    return table_dir


def _summary_directions(suite: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for entry in suite.get("summary_metrics", ()):
        result[str(entry["name"])] = str(entry.get("direction", "max"))
    return result


def _write_long_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    fields = [
        "problem",
        "method",
        "metric",
        "n",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
        "median",
        "min",
        "max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _safe_filename(metric: str) -> str:
    return metric.replace("/", "_").replace("\\", "_").replace(".", "_")


def _write_metric_tables(
    table_dir: Path,
    metric: str,
    summarized: Mapping[tuple[str, str, str], Mapping[str, Any]],
    direction: str,
) -> None:
    problems = sorted({problem for problem, _, item_metric in summarized if item_metric == metric})
    methods = sorted({method for _, method, item_metric in summarized if item_metric == metric})
    if not problems or not methods:
        return
    best: dict[str, list[str]] = {}
    for problem in problems:
        scored = [
            (method, summarized[(problem, method, metric)].get("mean"))
            for method in methods
            if (problem, method, metric) in summarized
            and summarized[(problem, method, metric)].get("mean") is not None
        ]
        scored.sort(key=lambda item: item[1], reverse=direction == "max")
        best[problem] = [method for method, _ in scored[:2]]

    safe = _safe_filename(metric)
    markdown = [f"# {metric}", "", "| Problem | " + " | ".join(methods) + " |"]
    markdown.append("|---|" + "|".join("---:" for _ in methods) + "|")
    latex = [
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        "Problem & " + " & ".join(methods) + " \\\\",
        "\\midrule",
    ]
    csv_path = table_dir / f"table_{safe}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(["Problem", *methods])
        for problem in problems:
            cells = []
            latex_cells = []
            for method in methods:
                stats = summarized.get((problem, method, metric), {})
                cell = _format_cell(stats)
                cells.append(cell)
                latex_cell = cell.replace("±", "$\\pm$")
                if best.get(problem, [None])[0:1] == [method]:
                    latex_cell = f"\\textbf{{{latex_cell}}}"
                elif method in best.get(problem, [])[1:2]:
                    latex_cell = f"\\underline{{{latex_cell}}}"
                latex_cells.append(latex_cell)
            writer.writerow([problem, *cells])
            markdown.append(f"| {problem} | " + " | ".join(cells) + " |")
            latex.append(problem.replace("_", "\\_") + " & " + " & ".join(latex_cells) + " \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (table_dir / f"table_{safe}.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (table_dir / f"table_{safe}.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir")
    parser.add_argument("--suite-config")
    args = parser.parse_args(argv)
    suite_config = {}
    if args.suite_config:
        with Path(args.suite_config).open("r", encoding="utf-8") as handle:
            suite_config = json.load(handle)
    output = summarize_suite(args.suite_dir, suite_config)
    print(f"Tables written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
