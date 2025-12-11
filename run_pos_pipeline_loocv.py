"""
run_pos_pipeline_loocv.py

LOOCV version of POS pipeline
For each topic, train personal/GPT POS PCFGs on other topics
rewrite held-out GPT essay toward the held-out personal style

files named with prefixes:
personal_<topic>.txt in --personal-dir
gpt_<topic>.txt in --gpt-dir
topic names align
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(
    cmd: List[str],
    log_file: Path,
    step_name: str,
    debug: bool = False,
    cwd: Optional[Path] = None,
):
    start = time.perf_counter()
    print(f"\n[STEP] {step_name}")
    print("  $", " ".join(cmd))
    if cwd:
        print(f"  (cwd={cwd})")

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n\n===== {step_name} =====\n")
        lf.write("CMD: " + " ".join(cmd) + "\n")
        if cwd:
            lf.write(f"cwd: {cwd}\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            lf.write(line + "\n")
            if debug:
                print("   ", line)

        proc.wait()
        elapsed = time.perf_counter() - start
        lf.write(f"\n[STEP COMPLETED] {step_name} in {elapsed:.2f}s\n")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {proc.returncode}: {' '.join(cmd)}"
        )
    print(f"[OK] {step_name} (t={elapsed:.2f}s)")


def timestamp_label(label: Optional[str]) -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{label}" if label else ts


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def concat_files(files: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for f in files:
            out.write(f.read_text(encoding="utf-8"))
            out.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="LOOCV POS pipeline"
    )
    parser.add_argument(
        "--personal-dir",
        type=str,
        required=True,
        help="Dir with personal_*.txt files",
    )
    parser.add_argument(
        "--gpt-dir",
        type=str,
        required=True,
        help="Dir with gpt_*.txt files",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="loocv",
        help="Label prefix for run dirs",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    personal_dir = Path(args.personal_dir).resolve()
    gpt_dir = Path(args.gpt_dir).resolve()
    if not personal_dir.exists() or not gpt_dir.exists():
        raise FileNotFoundError("Personal or GPT dir not found")

    #map topic -> paths
    personal_files = {p.stem.replace("personal_", "", 1): p for p in personal_dir.glob("personal_*.txt")}
    gpt_files = {p.stem.replace("gpt_", "", 1): p for p in gpt_dir.glob("gpt_*.txt")}
    topics = sorted(set(personal_files) & set(gpt_files))
    if not topics:
        raise ValueError("No overlapping topics found btwn personal_*.txt, gpt_*.txt")

    runs_root = Path("runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    for topic in topics:
        heldout_personal = personal_files[topic]
        heldout_gpt = gpt_files[topic]

        #training sets exclude held-out topic
        train_personal = [p for t, p in personal_files.items() if t != topic]
        train_gpt = [p for t, p in gpt_files.items() if t != topic]
        if not train_personal or not train_gpt:
            print(f"[SKIP] Not enough training data to hold out {topic}")
            continue

        run_name = timestamp_label(f"{args.label}_{topic}")
        run_dir = runs_root / run_name
        (run_dir / "models").mkdir(parents=True, exist_ok=True)
        (run_dir / "viz").mkdir(parents=True, exist_ok=True)
        (run_dir / "personal_data").mkdir(parents=True, exist_ok=True)
        (run_dir / "gpt_data").mkdir(parents=True, exist_ok=True)

        log_file = run_dir / "log.txt"

        #concat train corpora
        personal_train = run_dir / "personal_data" / "personal_train.txt"
        gpt_train = run_dir / "gpt_data" / "gpt_train.txt"
        concat_files(train_personal, personal_train)
        concat_files(train_gpt, gpt_train)

        #copy held-out raw for record
        shutil.copy2(heldout_personal, run_dir / "personal_data" / heldout_personal.name)
        shutil.copy2(heldout_gpt, run_dir / "gpt_data" / heldout_gpt.name)

        #preprocess training + held-out GPT
        personal_clean_jsonl = run_dir / "personal_data" / "personal_train_clean.jsonl"
        gpt_train_clean_jsonl = run_dir / "gpt_data" / "gpt_train_clean.jsonl"
        gpt_holdout_clean_jsonl = run_dir / "gpt_data" / f"{heldout_gpt.stem}_clean.jsonl"

        #run commands
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "process_sample_text.py"),
                "--in-dir",
                str(personal_train.parent),
                "--out-path",
                str(personal_clean_jsonl),
            ],
            log_file,
            f"Preprocess personal train (holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "process_sample_text.py"),
                "--in-dir",
                str(gpt_train.parent),
                "--out-path",
                str(gpt_train_clean_jsonl),
            ],
            log_file,
            f"Preprocess GPT train (holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "process_sample_text.py"),
                "--in-dir",
                str((run_dir / "gpt_data")),
                "--out-path",
                str(gpt_holdout_clean_jsonl),
            ],
            log_file,
            f"Preprocess GPT holdout (holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )

        #POS trees for train corpora
        personal_trees = run_dir / "personal_data" / "personal_train_pos_trees.jsonl"
        gpt_trees = run_dir / "gpt_data" / "gpt_train_pos_trees.jsonl"
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "make_POS_trees.py"),
                "--in-path",
                str(personal_clean_jsonl),
                "--out-path",
                str(personal_trees),
            ],
            log_file,
            f"Build POS trees (personal train, holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "make_POS_trees.py"),
                "--in-path",
                str(gpt_train_clean_jsonl),
                "--out-path",
                str(gpt_trees),
            ],
            log_file,
            f"Build POS trees (GPT train, holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )

        #POS PCFGs from train
        pcfg_personal = run_dir / "models" / "pcfg_personal_pos.json"
        pcfg_gpt = run_dir / "models" / "pcfg_gpt_pos.json"
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "pcfg_POS.py"),
                "--trees",
                str(personal_trees),
                "--out",
                str(pcfg_personal),
            ],
            log_file,
            f"Estimate POS PCFG (personal train, holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "pcfg_POS.py"),
                "--trees",
                str(gpt_trees),
                "--out",
                str(pcfg_gpt),
            ],
            log_file,
            f"Estimate POS PCFG (GPT train, holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )

        #baselines on train corpora
        baselines_out = run_dir / "models" / "style_baselines.json"
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "compute_baselines.py"),
                "--mine-train",
                str(personal_train.with_suffix(".txt")),
                "--gpt-train",
                str(gpt_train.with_suffix(".txt")),
                "--personal_grammar",
                str(pcfg_personal),
                "--gpt_grammar",
                str(pcfg_gpt),
                "--out-baselines",
                str(baselines_out),
            ],
            log_file,
            f"Compute baselines (train corpora, holdout={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )

        #rewrite held-out GPT using train grammars/baselines
        rewrite_out = run_dir / "rewritten_gpt_essay_relative_pos.txt"
        pos_change_log = run_dir / "models" / "pos_change_log.json"
        #pick cand file matching topic
        cand_path = None
        cand_name = Path(f"data/gpt_candidates_{topic}.jsonl")
        if cand_name.exists():
            cand_path = cand_name.resolve()
            ensure_dir(run_dir / "data")
            shutil.copy2(cand_path, run_dir / "data" / cand_path.name)
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "rewrite_POS.py"),
                "--mine",
                str(personal_train.with_suffix(".txt")),
                "--gpt",
                str(run_dir / "gpt_data" / heldout_gpt.name),
                "--you-grammar",
                str(pcfg_personal),
                "--gpt-grammar",
                str(pcfg_gpt),
                "--baselines",
                str(baselines_out),
                "--out-essay",
                str(rewrite_out),
                "--out-log",
                str(pos_change_log),
                *(
                    ["--cands", str(run_dir / "data" / cand_path.name)]
                    if cand_path is not None
                    else []
                ),
            ],
            log_file,
            f"Rewrite GPT holdout (topic={topic})",
            args.debug,
            cwd=SCRIPT_DIR,
        )

        print(f"[RUN COMPLETE] Holdout topic '{topic}' → {run_dir}")


if __name__ == "__main__":
    main()
