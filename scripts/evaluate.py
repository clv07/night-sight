"""Evaluate low-light methods A-E: image quality, ExDark detection, and runtime.

Run from the project root:
    python scripts/evaluate.py --split dev
    python scripts/evaluate.py --split dev --methods E --only detection
    python scripts/evaluate.py --split dev --datasets lol --only quality
    python scripts/evaluate.py --split dev --only runtime --runtime-images 10
    python scripts/evaluate.py --split test

Reads the outputs and results/low_light/params_<split>.json written by scripts/run_low_light.py
(run it first for the same split), data/splits/, and data/ExDark/labels/ (dataset/exdark.py).
Writes results/evaluation/<split>/:
    summary.md         tables for the report (per-method means, mAP@0.5 bootstrap CIs)
    summary.json       the same numbers + metadata per section and method (date, git, device,
                       method parameters, detector settings and weights hash, runtime size)
    lol_quality.csv    per (method, image): PSNR, SSIM vs reference + no-reference stats
    exdark_quality.csv per (method, image): no-reference stats
    detection.csv      per method: mAP@0.5 (+ 95% CI, difference vs A + CI), mAP@0.5:0.95,
                       precision, recall, per-class AP@0.5
    detection/<method>/per_image_stats.npz  per-image YOLO matches, reused for paired bootstraps
    runtime.csv        per (method, dataset): ms per image at a fixed input size
--split is required. Partial runs (--only / --methods / --datasets) replace just those entries and
keep the rest; a pre-refactor summary.json is not merged into but replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
from dataset.exdark import label_path  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    DETECT_STAT_KEYS, DETECTOR_WEIGHTS, RUNTIME_SIZE, RUNTIME_WARMUP, bootstrap_map50, default_device,
    detector_config, evaluate_detection, full_reference, no_reference, read_image, time_method,
)
from low_light.methods import METHOD_DIRS, build_methods, load_params, output_path, read_split  # noqa: E402

EVAL_ROOT = PROJECT_ROOT / "results" / "evaluation"
DATASETS = ("lol", "exdark")
SECTIONS = ("quality", "detection", "runtime")
SECTION_DATASETS = {"quality": ("lol", "exdark"), "detection": ("exdark",), "runtime": ("lol", "exdark")}
QUALITY_COLS = ["psnr", "ssim", "brightness", "entropy", "noise"]
# Detection: per-method keys in output order; *_ci entries are [low, high] lists.
DETECTION_COLS = ("map50", "map50_ci", "delta_vs_A", "delta_vs_A_ci", "map50_95", "precision", "recall")
STATS_FILE = "per_image_stats.npz"
# One shared resample set for every method, so differences between methods are paired.
BOOTSTRAP_SEED = 253
BOOTSTRAP_RESAMPLES = 2000


def mean_of(rows: list[dict], cols: list[str]) -> dict[str, float]:
    """Mean of each column in `cols` that the rows contain."""
    return {c: round(float(np.mean([r[c] for r in rows])), 6) for c in cols if c in rows[0]}


def merge(old: dict, new: dict) -> dict:
    """Recursively update `old` with `new`, keeping entries (e.g. other methods) that `new` lacks."""
    for k, v in new.items():
        old[k] = merge(old[k], v) if isinstance(v, dict) and isinstance(old.get(k), dict) else v
    return old


def section_methods(section: str, data: dict) -> list[str]:
    """Method names evaluated in one section's results (quality is keyed by dataset first)."""
    names = dict.fromkeys(n for ds in data.values() for n in ds) if section == "quality" else data
    return [n for n in names if n != "reference"]


def percentile_ci(values: np.ndarray) -> list[float]:
    """Percentile 95% interval [2.5th, 97.5th] of bootstrap values, rounded to 6 decimals."""
    return [round(float(x), 6) for x in np.percentile(values, [2.5, 97.5])]


def bootstrap_detection(stats: dict[str, dict[str, dict[str, np.ndarray]]], reference: str) -> dict[str, dict]:
    """Bootstrap CIs of each method's mAP@0.5 and of its difference from `reference`, over images.

    `stats` is {method name: per-image stats} with identical image keys. Every method is resampled
    with the same index set, so a difference's CI reflects only how the methods disagree per image.
    """
    images = sorted(next(iter(stats.values())))
    for name, per_image in stats.items():
        if sorted(per_image) != images:
            raise ValueError(f"{name} per-image stats cover different images than the other methods")
    n = len(images)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, (BOOTSTRAP_RESAMPLES, n))
    point = {name: float(bootstrap_map50(s, np.arange(n)[None])[0]) for name, s in stats.items()}
    boot = {name: bootstrap_map50(s, samples) for name, s in stats.items()}
    out: dict[str, dict] = {}
    for name in stats:
        out[name] = {"map50_ci": percentile_ci(boot[name])}
        if reference in stats and name != reference:
            out[name]["delta_vs_A"] = round(point[name] - point[reference], 6)
            out[name]["delta_vs_A_ci"] = percentile_ci(boot[name] - boot[reference])
    return out


def outputs_for(key: str, dataset: str, split: str, srcs: list[Path]) -> list[Path]:
    """Return the saved outputs of method `key` for `srcs`; exit with a hint if any are missing."""
    outs = [output_path(key, dataset, s) for s in srcs]
    missing = [o for o in outs if not o.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} {METHOD_DIRS[key]} {dataset} outputs missing (e.g. {missing[0]}).\n"
                         f"Run: python scripts/run_low_light.py --split {split} --datasets {dataset} --methods {key}")
    return outs


def quality_row(output: Path, reference: Path | None = None) -> dict:
    """Per-image quality metrics: no-reference stats, plus PSNR/SSIM when a reference is given."""
    image = read_image(output)
    metrics = {**(full_reference(image, read_image(reference)) if reference else {}), **no_reference(image)}
    name = output.name if reference else f"{output.parent.name}/{output.name}"
    return {"image": name, **{k: round(v, 6) for k, v in metrics.items()}}


def write_csv(path: Path, rows: list[dict], keep_existing: bool = False) -> None:
    """Write rows; with keep_existing, first keep earlier rows for methods not present in `rows`."""
    if keep_existing and path.exists():
        new_methods = {r["method"] for r in rows}
        with path.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["method"] not in new_methods] + rows
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union of keys, first-seen order
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def save_detection_stats(path: Path, per_image: dict[str, dict[str, np.ndarray]]) -> None:
    """Save per-image detection stats as flat arrays plus per-image counts (no pickled objects)."""
    names = sorted(per_image)
    arrays = {k: np.concatenate([per_image[n][k] for n in names]) for k in DETECT_STAT_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("wb") as f:  # a file object stops numpy from appending ".npz"
            np.savez_compressed(f, images=np.array(names), **arrays,
                                n_pred=np.array([len(per_image[n]["conf"]) for n in names]),
                                n_target=np.array([len(per_image[n]["target_cls"]) for n in names]))
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def load_detection_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load per-image detection stats written by save_detection_stats."""
    with np.load(path) as z:
        pred_cuts, target_cuts = np.cumsum(z["n_pred"])[:-1], np.cumsum(z["n_target"])[:-1]
        parts = {k: np.split(z[k], target_cuts if k == "target_cls" else pred_cuts) for k in DETECT_STAT_KEYS}
        return {str(name): {k: parts[k][i] for k in DETECT_STAT_KEYS} for i, name in enumerate(z["images"])}


def run_metadata(split: str) -> dict:
    """Run-level metadata: date, git commit and dirty state, device, and library versions."""
    def git(*cmd: str) -> str:
        """Run a git command in the project root; return stdout or "" if git is unavailable."""
        try:
            return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "src", "scripts", "dataset")),
        "device": default_device(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def run_quality(keys: list[str], split: str, datasets: list[str]) -> tuple[dict[str, list[dict]], dict]:
    """Per-image and mean quality metrics: LOL against its references, ExDark no-reference only."""
    srcs = {ds: read_split(ds, split) for ds in datasets}
    lol_refs = [s.parents[1] / "high" / s.name for s in srcs.get("lol", [])]
    per_image: dict[str, list[dict]] = {ds: [] for ds in datasets}
    summary: dict[str, dict] = {ds: {} for ds in datasets}
    for key in keys:
        name = METHOD_DIRS[key]
        for ds in datasets:
            outs = outputs_for(key, ds, split, srcs[ds])
            refs = lol_refs if ds == "lol" else [None] * len(outs)
            rows = [quality_row(o, r) for o, r in zip(outs, refs)]
            per_image[ds] += [{"method": name, **r} for r in rows]
            summary[ds][name] = mean_of(rows, QUALITY_COLS)
        shown = []
        if "lol" in summary:
            shown.append(f"LOL PSNR {summary['lol'][name]['psnr']:.2f}  SSIM {summary['lol'][name]['ssim']:.3f}")
        if "exdark" in summary:
            shown.append(f"ExDark brightness {summary['exdark'][name]['brightness']:.3f}")
        print(f"quality   {name:<15} " + "  ".join(shown))
    if "lol" in summary:
        # The references themselves, as a target for the no-reference stats.
        summary["lol"]["reference"] = mean_of([quality_row(r) for r in lol_refs], QUALITY_COLS)
    return per_image, summary


def run_detection(keys: list[str], split: str, out_dir: Path, weights: Path) -> dict:
    """Frozen-YOLOv8 detection metrics on each method's ExDark outputs, with paired bootstrap CIs.

    Each method's per-image stats are saved, so a later partial run (e.g. --methods E) can still
    report its difference from A without re-validating A.
    """
    srcs = read_split("exdark", split)
    labels = [label_path(s) for s in srcs]
    results: dict[str, dict] = {}
    stats: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for key in keys:
        name = METHOD_DIRS[key]
        results[name], stats[name] = evaluate_detection(outputs_for(key, "exdark", split, srcs), labels,
                                                        out_dir / "detection" / name, weights=weights)
        save_detection_stats(out_dir / "detection" / name / STATS_FILE, stats[name])
        print(f"detection {name:<15} mAP50 {results[name]['map50']:.3f}  mAP50-95 {results[name]['map50_95']:.3f}")

    reference = METHOD_DIRS["A"]
    reference_path = out_dir / "detection" / reference / STATS_FILE
    if reference not in stats:
        stored = load_detection_stats(reference_path) if reference_path.exists() else {}
        if stored and sorted(stored) == sorted(next(iter(stats.values()))):
            stats[reference] = stored
        else:
            print(f"No delta_vs_A: {reference_path} is missing or covers other images; add --methods A")
    print(f"bootstrap {BOOTSTRAP_RESAMPLES} resamples x {len(stats)} methods (seed {BOOTSTRAP_SEED})")
    boot = bootstrap_detection(stats, reference)
    for name in results:
        merged = {**results[name], **boot[name]}
        results[name] = {k: merged[k] for k in (*DETECTION_COLS, "per_class") if k in merged}
        lo, hi = merged["map50_ci"]
        delta = (f"  vs A {merged['delta_vs_A']:+.3f} [{merged['delta_vs_A_ci'][0]:+.3f}, "
                 f"{merged['delta_vs_A_ci'][1]:+.3f}]" if "delta_vs_A" in merged else "")
        print(f"bootstrap {name:<15} mAP50 {merged['map50']:.3f} [{lo:.3f}, {hi:.3f}]{delta}")
    return results


def run_runtime(keys: list[str], split: str, datasets: list[str], n_images: int) -> dict:
    """Time each method per dataset at RUNTIME_SIZE with the parameters that produced its outputs."""
    results: dict[str, dict] = {}
    for dataset in datasets:
        srcs = read_split(dataset, split)
        srcs = srcs[:: max(1, len(srcs) // n_images)][:n_images]  # evenly spaced subset, all classes
        images = [read_image(p) for p in srcs]
        for key in keys:
            if key == "A":  # identity copy: nothing to time
                continue
            name = METHOD_DIRS[key]
            fn = build_methods([key], **load_params(split, dataset, key))[key]
            results.setdefault(name, {})[dataset] = time_method(fn, images)
            print(f"runtime   {name:<15} {dataset:<6} {results[name][dataset]['ms_mean']:.1f} ms/img")
    return results


def section_metadata(section: str, names: list[str], split: str, datasets: list[str], base: dict,
                     weights: Path) -> dict:
    """Metadata for each method of one section: run info, method parameters, section settings."""
    extra = {}
    if section == "detection":
        extra["detector"] = detector_config(weights)
        extra["bootstrap"] = {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                              "unit": "image", "interval": "percentile 95%", "paired": True}
    elif section == "runtime":
        extra["runtime"] = {"width": RUNTIME_SIZE[0], "height": RUNTIME_SIZE[1], "warmup": RUNTIME_WARMUP}
    key_of = {v: k for k, v in METHOD_DIRS.items()}
    return {name: {**base, **extra, "params": {ds: load_params(split, ds, key_of[name]) for ds in datasets}}
            for name in names}


def cell(value: float | list[float], fmt: str) -> str:
    """Format one table value; a list (a [low, high] CI) becomes "[low, high]"."""
    return "[" + ", ".join(format(x, fmt) for x in value) + "]" if isinstance(value, list) else format(value, fmt)


def table(title: str, rows: dict[str, dict], cols: list[tuple[str, str, str]]) -> str:
    """Markdown table: rows {method: {col: value}}, cols [(key, header, format)]."""
    lines = [f"## {title}", "", "| Method | " + " | ".join(h for _, h, _ in cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for method, vals in sorted(rows.items(), key=lambda kv: (kv[0] == "reference", kv[0])):
        lines.append(f"| {method} | " + " | ".join(cell(vals[k], f) if k in vals else "–" for k, _, f in cols) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, base: dict, results: dict) -> None:
    """Write summary.md with one table per evaluated section; per-method metadata is in summary.json."""
    meta = results.get("metadata", {})
    parts = [f"# Low-light evaluation ({base['split']} split)", "",
             f"Last updated {base['date']} · commit `{base['git_commit']}`"
             f"{' (uncommitted changes)' if base['git_dirty'] else ''} · {base['device']} · torch {base['torch']}."
             " Per-section run details are in summary.json.", ""]
    q = results.get("quality", {})
    if "lol" in q:
        parts.append(table("LOL: image quality (vs normal-light reference)", q["lol"], [
            ("psnr", "PSNR (dB) ↑", ".2f"), ("ssim", "SSIM ↑", ".3f"), ("brightness", "Brightness", ".3f"),
            ("entropy", "Entropy (bits)", ".2f"), ("noise", "Noise σ ↓", ".2f")]))
    if "exdark" in q:
        parts.append(table("ExDark: no-reference image statistics", q["exdark"], [
            ("brightness", "Brightness", ".3f"), ("entropy", "Entropy (bits)", ".2f"), ("noise", "Noise σ ↓", ".2f")]))
    d = results.get("detection")
    if d:
        detectors = sorted({m["detector"]["model"] for m in meta.get("detection", {}).values() if "detector" in m})
        title = f"ExDark: detection ({', '.join(detectors) or 'YOLO'}, COCO-pretrained, frozen)"
        parts.append(table(title, d, [
            ("map50", "mAP@0.5 ↑", ".3f"), ("map50_ci", "95% CI", ".3f"), ("delta_vs_A", "Δ vs A", "+.3f"),
            ("delta_vs_A_ci", "Δ 95% CI", "+.3f"), ("map50_95", "mAP@0.5:0.95 ↑", ".3f"),
            ("precision", "Precision", ".3f"), ("recall", "Recall", ".3f")]))
        parts.append(f"CIs: percentile bootstrap over images ({BOOTSTRAP_RESAMPLES} resamples, seed "
                     f"{BOOTSTRAP_SEED}), one resample set shared by all methods, so Δ vs A is paired.\n")
        classes = sorted({c for v in d.values() for c in v["per_class"]})
        parts.append(table("ExDark: AP@0.5 per class", {m: v["per_class"] for m, v in d.items()},
                           [(c, c, ".3f") for c in classes]))
    r = results.get("runtime")
    if r:
        flat = {m: {f"{ds}_{k}": x for ds, s in v.items() for k, x in s.items()} for m, v in r.items()}
        parts.append(table(f"Runtime per image at {RUNTIME_SIZE[0]}×{RUNTIME_SIZE[1]} ({base['device']})", flat, [
            ("lol_ms_mean", "LOL ms", ".1f"), ("exdark_ms_mean", "ExDark ms", ".1f"),
            ("exdark_ms_median", "ExDark median ms", ".1f")]))
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved: {path}")


def detection_csv_row(method: str, result: dict) -> dict:
    """Flatten one method's detection result for detection.csv: CI lists become _lo/_hi columns."""
    row: dict = {"method": method}
    for k in DETECTION_COLS:
        if k.endswith("_ci"):
            row[f"{k}_lo"], row[f"{k}_hi"] = result.get(k, ("", ""))
        else:
            row[k] = result.get(k, "")
    return {**row, **{f"ap50_{c}": ap for c, ap in result["per_class"].items()}}


def main() -> None:
    """Parse arguments, run the requested sections, and merge them into the split's records."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_DIRS), choices=list(METHOD_DIRS))
    parser.add_argument("--only", nargs="+", default=list(SECTIONS), choices=SECTIONS)
    parser.add_argument("--weights", type=Path, default=DETECTOR_WEIGHTS, help="YOLO weights for detection")
    parser.add_argument("--runtime-images", type=int, default=50, help="Images per dataset for timing")
    args = parser.parse_args()
    if args.runtime_images < 1:
        parser.error("--runtime-images must be >= 1")
    datasets = {s: [d for d in SECTION_DATASETS[s] if d in args.datasets] for s in args.only}
    for section, names in datasets.items():
        if not names:
            print(f"Skipping {section}: it needs --datasets {' '.join(SECTION_DATASETS[section])}")
    sections = [s for s in args.only if datasets[s]]
    if not sections:
        parser.error(f"--only {' '.join(args.only)} has nothing to evaluate for --datasets {' '.join(args.datasets)}")
    # Fail before any heavy work if a method's outputs were not produced by run_low_light.py.
    try:
        for section in sections:
            for dataset in datasets[section]:
                for key in args.methods:
                    load_params(args.split, dataset, key)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    out_dir = EVAL_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    results = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    # The pre-refactor format kept one flat run record ("updated", metadata.date, ...). Its numbers came
    # from an older protocol, so start a fresh record instead of merging, and drop its CSV rows too.
    legacy = "updated" in results or any(s not in SECTIONS for s in results.get("metadata", {}))
    if legacy:
        print(f"Warning: {summary_path} has the pre-refactor format; starting a fresh record. "
              "CSVs of sections not re-run now are stale.")
        results = {}

    new: dict[str, dict] = {}
    if "quality" in sections:
        per_image, new["quality"] = run_quality(args.methods, args.split, datasets["quality"])
        for dataset in datasets["quality"]:
            write_csv(out_dir / f"{dataset}_quality.csv", per_image[dataset], keep_existing=not legacy)
    if "detection" in sections:
        new["detection"] = run_detection(args.methods, args.split, out_dir, args.weights)
    if "runtime" in sections:
        new["runtime"] = run_runtime(args.methods, args.split, datasets["runtime"], args.runtime_images)

    base = run_metadata(args.split)
    new["metadata"] = {s: section_metadata(s, section_methods(s, new[s]), args.split, datasets[s], base, args.weights)
                       for s in sections}
    # Replace whole detection entries, so keys a re-run no longer produces (old speed_ms, a delta_vs_A
    # that could not be recomputed) do not linger; other sections merge per dataset.
    for name in new.get("detection", {}):
        results.get("detection", {}).pop(name, None)
    merge(results, new)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")

    if "detection" in results:
        write_csv(out_dir / "detection.csv", [detection_csv_row(m, r) for m, r in results["detection"].items()])
    if "runtime" in results:
        write_csv(out_dir / "runtime.csv", [
            {"method": m, "dataset": ds, **s} for m, v in results["runtime"].items() for ds, s in v.items()])
    write_markdown(out_dir / "summary.md", base, results)


if __name__ == "__main__":
    main()
