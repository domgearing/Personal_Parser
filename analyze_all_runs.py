"""
analyze_all_runs.py

Aggregate results from pipeline runs - under runs/.

For each run dir. (runs/.../):

-loads:
    - style_stats_pos.json
    - style_baselines.json
    - pos_change_log.json
-calculates per run summary stats:
    - # sents in pos_change_log
    - # sents changed (orig != new)
    - counts of added/rem POS bigrams
-compiles across runs:
    -row per run in runs_summary.csv
    -tot counts of added/rem POS bigrams across runs
      in pos_bigram_changes_all_runs.csv

command to run:
    python analyze_all_runs.py --runs-root runs --out-dir runs_summary
"""

import json
import pathlib
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple

import pandas as pd


def safe_load_json(path: pathlib.Path) -> Dict[str, Any] | List[Any] | None:
    ##Load JSON file##
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load JSON from {path}: {e}")
        return None


def summarize_pos_change_log(path: pathlib.Path) -> dict:
    """
    summarize pos_change_log.json 

    pos_change_log.json structure:
        [
          {
            "sent_idx": int,
            "orig": str,
            "new": str,
            "added": [["DT","NN"], ...],
            "removed": [["IN","DT"], ...]
          }
        ]
    """
    data = safe_load_json(path)
    if data is None:
        return {
            "n_sentences_in_log": 0,
            "n_sentences_changed": 0,
            "added_bigram_counts": Counter(),
            "removed_bigram_counts": Counter(),
        }

    n_sent = len(data)
    n_changed = 0
    added_counter: Counter[Tuple[str, str]] = Counter()
    removed_counter: Counter[Tuple[str, str]] = Counter()

    for rec in data:
        orig = rec.get("orig", "")
        new = rec.get("new", "")
        if orig != new:
            n_changed += 1

        added = rec.get("added", [])
        removed = rec.get("removed", [])

        # each bit in add/rem ex: ["DT", "NN"]
        for pair in added:
            if isinstance(pair, list) and len(pair) == 2:
                added_counter[(pair[0], pair[1])] += 1
        for pair in removed:
            if isinstance(pair, list) and len(pair) == 2:
                removed_counter[(pair[0], pair[1])] += 1

    return {
        "n_sentences_in_log": n_sent,
        "n_sentences_changed": n_changed,
        "added_bigram_counts": added_counter,
        "removed_bigram_counts": removed_counter,
    }


def flatten_style_stats(style_stats: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    flatten style_stats_pos.json into flat dict
    top-level keys exist
    """
    if style_stats is None:
        return {}
    flat: Dict[str, Any] = {}
    for k, v in style_stats.items():
        #if nested dict, flatten one level
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Combine multi run PCFG style/POS-change results."
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root dir. cont. run subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs_summary",
        help="dir. to write agg. outs.",
    )
    args = parser.parse_args()

    runs_root = pathlib.Path(args.runs_root)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [d for d in runs_root.iterdir() if d.is_dir()]
    run_dirs = sorted(run_dirs)

    print(f"[INFO] Found {len(run_dirs)} run directories under {runs_root}")

    rows: List[Dict[str, Any]] = []

    #glob POS bigram counts - all runs
    global_added: Counter[Tuple[str, str]] = Counter()
    global_removed: Counter[Tuple[str, str]] = Counter()

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"\n[INFO] Processing run: {run_name}")

        #load per-run JSONs
        models_dir = run_dir / "models"
        style_stats_path = models_dir / "style_stats_pos.json"
        baselines_path = models_dir / "style_baselines.json"
        pos_change_log_path = models_dir / "pos_change_log.json"

        style_stats = safe_load_json(style_stats_path)
        baselines = safe_load_json(baselines_path)
        pos_summary = summarize_pos_change_log(pos_change_log_path)

        #update glob bigram counters
        global_added.update(pos_summary["added_bigram_counts"])
        global_removed.update(pos_summary["removed_bigram_counts"])

        #flatten style stats, baselines to one row
        row: Dict[str, Any] = {"run": run_name}

        #flatten style_stats_pos.json
        row.update(flatten_style_stats(style_stats))

        #add baselines
        if isinstance(baselines, dict):
            for k, v in baselines.items():
                row[f"baseline.{k}"] = v

        #add rewrite stats
        row["n_sentences_in_log"] = pos_summary["n_sentences_in_log"]
        row["n_sentences_changed"] = pos_summary["n_sentences_changed"]
        row["prop_sentences_changed"] = (
            pos_summary["n_sentences_changed"] / pos_summary["n_sentences_in_log"]
            if pos_summary["n_sentences_in_log"] > 0
            else 0.0
        )

        rows.append(row)

    #save per run sum. table
    if rows:
        df = pd.DataFrame(rows).sort_values("run")
        summary_csv = out_dir / "runs_summary.csv"
        df.to_csv(summary_csv, index=False)
        print(f"\n[SAVED] Per-run summary CSV -> {summary_csv}")
    else:
        print("[WARN] No runs processed; no rows to save.")

    #save glob POS bigram change count
    bigram_rows: List[Dict[str, Any]] = []
    all_keys = set(global_added.keys()) | set(global_removed.keys())
    for big in sorted(all_keys):
        added_count = global_added.get(big, 0)
        removed_count = global_removed.get(big, 0)
        bigram_rows.append(
            {
                "pos_bigram_from": big[0],
                "pos_bigram_to": big[1],
                "added_count": added_count,
                "removed_count": removed_count,
                "net_change": added_count - removed_count,
            }
        )

    if bigram_rows:
        df_bigrams = pd.DataFrame(bigram_rows).sort_values("net_change", ascending=False)
        bigram_csv = out_dir / "pos_bigram_changes_all_runs.csv"
        df_bigrams.to_csv(bigram_csv, index=False)
        print(f"[SAVED] Global POS bigram change counts -> {bigram_csv}")
    else:
        print("[WARN] No POS bigram changes recorded across runs.")


if __name__ == "__main__":
    main()
