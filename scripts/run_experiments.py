"""Batch runner for PROBE cross-domain experiments.

Reads the domain registry (data/domains.json), iterates over all source→target
pairs, runs the full PROBE pipeline for each, and logs results to CSV.

Usage
-----
  # Dry run: print what would be executed
  python scripts/run_experiments.py --dry-run

  # Run a single pair
  python scripts/run_experiments.py --source china_motorbike --target japan --device cuda

  # Run all cross-domain pairs
  python scripts/run_experiments.py --device cuda

  # Skip pretraining (reuse existing pretrained checkpoints)
  python scripts/run_experiments.py --device cuda --skip-pretrain
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_registry(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_domain(registry: list[dict], name: str) -> dict:
    for d in registry:
        if d["name"] == name:
            return d
    raise ValueError(f"Domain '{name}' not found in registry. Available: {[d['name'] for d in registry]}")


def generate_experiment_config(
    source: dict,
    target: dict,
    base_config: Path,
    output_dir: Path,
    pretrain_epochs: int = 100,
    det_epochs: int = 50,
) -> Path:
    """Generate a YAML config for a specific source→target experiment pair."""
    import yaml

    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pair_name = f"{source['name']}_to_{target['name']}"

    # Override data paths
    data_dir = Path(source["image_root"]).parent  # assumes everything under data/
    cfg["data"]["source_manifest"] = str(data_dir / source["train_manifest"])
    cfg["data"]["target_manifest"] = str(data_dir / target["unlabeled_manifest"])
    cfg["data"]["val_manifest"] = str(data_dir / target["val_manifest"])
    cfg["data"]["image_root"] = source["image_root"]

    # Override training epochs
    cfg["optim"]["pretrain_epochs"] = pretrain_epochs
    cfg["detection_optim"]["epochs"] = det_epochs

    # Write config
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"exp_{pair_name}.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return config_path


def run_command(cmd: list[str], log_file: Optional[Path] = None) -> int:
    """Run a shell command, optionally logging output to a file."""
    print(f"  RUN: {' '.join(cmd)}")
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        return result.returncode
    else:
        result = subprocess.run(cmd)
        return result.returncode


def parse_mAP_from_log(log_path: Path) -> Optional[float]:
    """Extract mAP@0.5 value from an inference log file."""
    if not log_path.exists():
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "mAP@0.5:" in line:
                try:
                    return float(line.split("mAP@0.5:")[1].strip().split()[0])
                except (ValueError, IndexError):
                    continue
    return None


def parse_class_APs_from_log(log_path: Path) -> dict[str, float]:
    """Extract per-class AP values from inference log."""
    aps = {}
    if not log_path.exists():
        return aps
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "AP_cls_" in line or "D0" in line or "D1" in line or "D4" in line or "Repair" in line:
                # Parse lines like "D00: Longitudinal Crack: 0.4523"
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    try:
                        val = float(parts[-1].strip())
                        key = parts[0].strip().replace(" ", "_")
                        aps[key] = val
                    except ValueError:
                        pass
    return aps


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    source: dict,
    target: dict,
    args: argparse.Namespace,
    results_dir: Path,
) -> dict:
    """Run the full PROBE pipeline for one source→target pair.

    Returns a dict of results for CSV logging.
    """
    pair_name = f"{source['name']}_to_{target['name']}"
    exp_dir = results_dir / pair_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {pair_name}")
    print(f"  source: {source['name']} ({source['num_labeled']} labeled)")
    print(f"  target: {target['name']} ({target['num_unlabeled']} unlabeled, {target['num_val']} val)")
    print(f"  output: {exp_dir}")
    print(f"{'='*70}")

    timestamp = datetime.now().isoformat()

    # Generate experiment config
    config_path = generate_experiment_config(
        source, target,
        base_config=Path(args.config),
        output_dir=exp_dir,
        pretrain_epochs=args.pretrain_epochs,
        det_epochs=args.det_epochs,
    )

    pretrain_ckpt = exp_dir / "checkpoints" / "probe_final.pt"
    det_ckpt = exp_dir / "checkpoints" / "probe_det_best.pt"

    if not args.dry_run:
        # --- Step 1: Pretraining (Phase 1 + 2) ---
        if not args.skip_pretrain:
            print("\n[1/3] SSL Pretraining...")
            train_cmd = [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "train.py"),
                "--config", str(config_path),
                "--device", args.device,
                "--epochs", str(args.pretrain_epochs),
                "--checkpoint-dir", str(exp_dir / "checkpoints"),
                "--no-viz",
            ]
            if args.batch_size:
                train_cmd += ["--batch-size", str(args.batch_size)]
            ret = run_command(train_cmd, log_file=exp_dir / "pretrain.log")
            if ret != 0:
                print(f"  ERROR: pretraining failed (exit {ret}) — see {exp_dir / 'pretrain.log'}")
                return {
                    "source": source["name"],
                    "target": target["name"],
                    "status": "pretrain_failed",
                    "exit_code": ret,
                    "timestamp": timestamp,
                }
        else:
            print("\n[1/3] SSL Pretraining — SKIPPED (using existing checkpoint)")

        # --- Step 2: Detection head training (Phase 3) ---
        if pretrain_ckpt.exists() or args.skip_pretrain:
            ckpt_to_use = pretrain_ckpt if pretrain_ckpt.exists() else args.resume_checkpoint
            if ckpt_to_use is None:
                print("  ERROR: no pretrained checkpoint found and --resume-checkpoint not set")
                return {
                    "source": source["name"],
                    "target": target["name"],
                    "status": "no_checkpoint",
                    "timestamp": timestamp,
                }
            print(f"\n[2/3] Detection Head Training...")
            det_cmd = [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "train.py"),
                "--config", str(config_path),
                "--phase", "3",
                "--resume", str(ckpt_to_use),
                "--device", args.device,
                "--det-epochs", str(args.det_epochs),
                "--checkpoint-dir", str(exp_dir / "checkpoints"),
            ]
            if args.batch_size:
                det_cmd += ["--batch-size", str(args.batch_size)]
            ret = run_command(det_cmd, log_file=exp_dir / "detection.log")
            if ret != 0:
                print(f"  ERROR: detection training failed (exit {ret}) — see {exp_dir / 'detection.log'}")
                return {
                    "source": source["name"],
                    "target": target["name"],
                    "status": "detection_failed",
                    "exit_code": ret,
                    "timestamp": timestamp,
                }
        else:
            print(f"  ERROR: pretrained checkpoint not found at {pretrain_ckpt}")
            return {
                "source": source["name"],
                "target": target["name"],
                "status": "no_checkpoint",
                "timestamp": timestamp,
            }

        # --- Step 3: Evaluation ---
        eval_ckpt = det_ckpt if det_ckpt.exists() else (pretrain_ckpt if pretrain_ckpt.exists() else None)
        if eval_ckpt is None:
            print("  ERROR: no checkpoint for evaluation")
            return {
                "source": source["name"],
                "target": target["name"],
                "status": "no_checkpoint",
                "timestamp": timestamp,
            }

        print(f"\n[3/3] Evaluation (mAP)...")
        eval_cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "infer.py"),
            "--config", str(config_path),
            "--checkpoint", str(eval_ckpt),
            "--device", args.device,
            "--eval",
        ]
        eval_log = exp_dir / "eval.log"
        ret = run_command(eval_cmd, log_file=eval_log)
        if ret != 0:
            print(f"  ERROR: evaluation failed (exit {ret}) — see {eval_log}")
            return {
                "source": source["name"],
                "target": target["name"],
                "status": "eval_failed",
                "exit_code": ret,
                "timestamp": timestamp,
            }

        mAP = parse_mAP_from_log(eval_log) or 0.0
        class_aps = parse_class_APs_from_log(eval_log)

        result = {
            "source": source["name"],
            "target": target["name"],
            "status": "ok",
            "mAP@0.5": mAP,
            "pretrain_epochs": args.pretrain_epochs,
            "det_epochs": args.det_epochs,
            "timestamp": timestamp,
        }
        result.update(class_aps)
        return result

    else:
        # Dry run: just print what would happen
        print(f"  [DRY RUN] Would run:")
        print(f"    1. Pretrain: train.py --config {config_path} --device {args.device}")
        print(f"    2. Det head: train.py --phase 3 --resume {pretrain_ckpt}")
        print(f"    3. Evaluate: infer.py --eval --checkpoint {det_ckpt}")
        return {
            "source": source["name"],
            "target": target["name"],
            "status": "dry_run",
            "timestamp": timestamp,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner for PROBE cross-domain experiments"
    )
    parser.add_argument(
        "--config", default="configs/probe_base.yaml",
        help="Base YAML config (data paths will be overridden per experiment)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--det-epochs", type=int, default=50)
    parser.add_argument("--results-dir", default="experiments")
    parser.add_argument("--registry", default="data/domains.json")
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Skip Phase 1+2 (use existing checkpoints)")
    parser.add_argument("--resume-checkpoint", default=None,
                        help="Path to pretrained checkpoint for --skip-pretrain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment matrix without executing")

    # Filter options
    parser.add_argument("--source", default=None, help="Run only this source domain")
    parser.add_argument("--target", default=None, help="Run only this target domain")
    args = parser.parse_args()

    # Load domain registry
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"ERROR: domain registry not found: {registry_path}")
        print("Run 'python scripts/prepare_multidomain.py' first.")
        sys.exit(1)

    registry = load_registry(registry_path)
    print(f"Loaded {len(registry)} domains: {[d['name'] for d in registry]}")

    # Build experiment list
    if args.source and args.target:
        pairs = [(find_domain(registry, args.source), find_domain(registry, args.target))]
    elif args.source:
        src = find_domain(registry, args.source)
        pairs = [(src, tgt) for tgt in registry if tgt["name"] != args.source]
    elif args.target:
        tgt = find_domain(registry, args.target)
        pairs = [(src, tgt) for src in registry if src["name"] != args.target]
    else:
        pairs = [(s, t) for s in registry for t in registry if s["name"] != t["name"]]

    print(f"Experiments: {len(pairs)} source->target pairs\n")

    # Run experiments
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "results.csv"
    all_results = []

    for source, target in pairs:
        result = run_single_experiment(source, target, args, results_dir)
        all_results.append(result)

        # Write CSV incrementally
        if result and result.get("status") not in ("dry_run",):
            fieldnames = list(result.keys())
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
            print(f"  -> CSV updated: {csv_path}")

    # Summary
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    if not args.dry_run:
        successful = [r for r in all_results if r.get("status") == "ok"]
        failed = [r for r in all_results if r.get("status") != "ok"]
        print(f"  Success: {len(successful)}")
        print(f"  Failed:  {len(failed)}")
        if successful:
            maps = [r["mAP@0.5"] for r in successful]
            print(f"  mAP range: [{min(maps):.4f}, {max(maps):.4f}]")
            print(f"  mAP mean:  {sum(maps)/len(maps):.4f}")
    print(f"  Results: {csv_path}")


if __name__ == "__main__":
    main()
