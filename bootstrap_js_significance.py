"""
bootstrap_js_significance.py

Bootstrap significance test for JS divergence between personal POS PCFG
and rewritten GPT corpus. 

Builds random "sample" essays by choosing randomly for each sent in GPT text, either original sent or one of
3 candidate rewrites. 

for each bootstrap sample, builds POS PCFG, computes JS(personal || sample). 

compares observed JS to this bootstrap dist and reports p-value.
"""

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: List[str], cwd: Path):
    if cmd and cmd[0].endswith(".py"):
        cmd = [sys.executable] + cmd
    subprocess.run(cmd, check=True, cwd=str(cwd))


def split_sents(text: str) -> List[str]:
    #sent splitter 
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_candidates(path: Path) -> Dict[str, List[str]]:
    """
    load JSONL with fields: orig, cands
    return map orig_sentence -> list of candidates.
    """
    out: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            orig = rec.get("orig", "").strip()
            cands = rec.get("cands", [])
            if orig:
                out[orig] = cands
    return out


def sample_corpus(sentences: Sequence[str], cand_map: Dict[str, List[str]]) -> str:
    """
    randomly pick either original or one of candidates per sent
    """
    chosen = []
    for s in sentences:
        options = [s] + cand_map.get(s, [])
        chosen.append(random.choice(options))
    return " ".join(chosen)


def build_pcfg_from_text(text: str, workdir: Path) -> Path:
    """
    write text to workdir/raw.txt, run process_sample_text -> make_POS_trees -> pcfg_POS
    return path to PCFG JSON.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    raw = workdir / "raw.txt"
    raw.write_text(text, encoding="utf-8")
    clean_jsonl = workdir / "raw_clean.jsonl"
    trees = workdir / "raw_pos_trees.jsonl"
    pcfg_out = workdir / "raw_pcfg_pos.json"

    run_cmd(
        [
            str(SCRIPT_DIR / "process_sample_text.py"),
            "--in-dir",
            str(workdir),
            "--out-path",
            str(clean_jsonl),
        ],
        cwd=SCRIPT_DIR,
    )
    run_cmd(
        [
            str(SCRIPT_DIR / "make_POS_trees.py"),
            "--in-path",
            str(clean_jsonl),
            "--out-path",
            str(trees),
        ],
        cwd=SCRIPT_DIR,
    )
    run_cmd(
        [
            str(SCRIPT_DIR / "pcfg_POS.py"),
            "--trees",
            str(trees),
            "--out",
            str(pcfg_out),
        ],
        cwd=SCRIPT_DIR,
    )
    return pcfg_out


def flatten_pcfg(pcfg_path: Path) -> Dict[str, float]:
    data = json.loads(pcfg_path.read_text(encoding="utf-8"))
    flat = {}
    for lhs, rules in data.items():
        for r in rules:
            rhs = " ".join(r["rhs"])
            flat[f"{lhs} -> {rhs}"] = r["prob"]
    return flat


def jensen_shannon(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-9) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a, b):
        s = 0.0
        for k in keys:
            pa = a.get(k, 0.0) + eps
            pb = b.get(k, 0.0) + eps
            s += pa * math.log(pa / pb)
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    parser = argparse.ArgumentParser(
        description="bootstrap JS sig. btwn personal PCFG, rewritten GPT PCFG"
    )
    parser.add_argument(
        "--personal-pcfg",
        type=str,
        required=True,
        help="path to personal PCFG JSON",
    )
    parser.add_argument(
        "--gpt-text",
        type=str,
        required=True,
        help="path to orig. GPT text cleaned",
    )
    parser.add_argument(
        "--cands",
        type=str,
        required=True,
        help="path to GPT cand. JSONL (orig + cands)",
    )
    parser.add_argument(
        "--observed-js",
        type=float,
        required=True,
        help="Observed JS(personal || rewritten) to test against",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="# bootstrap samples (default - 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Rand seed",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="bootstrap_js_results.txt",
        help="output summary file",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    personal_flat = flatten_pcfg(Path(args.personal_pcfg).resolve())
    gpt_text = Path(args.gpt_text).read_text(encoding="utf-8")
    sentences = split_sents(gpt_text)
    cand_map = load_candidates(Path(args.cands).resolve())

    bootstrap_js: List[float] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="bootstrap_js_"))
    try:
        for i in range(args.runs):
            sample_text = sample_corpus(sentences, cand_map)
            workdir = tmp_root / f"sample_{i}"
            sample_pcfg = build_pcfg_from_text(sample_text, workdir)
            sample_flat = flatten_pcfg(sample_pcfg)
            js = jensen_shannon(personal_flat, sample_flat)
            bootstrap_js.append(js)
            print(f"[BOOT {i+1}/{args.runs}] JS={js:.6f}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    obs = args.observed_js
    #two-sided: prop of bootstrap JS <= obs (lower = better here)
    p_value = sum(1 for x in bootstrap_js if x <= obs) / len(bootstrap_js)

    summary_lines = [
        f"Personal PCFG: {args.personal_pcfg}",
        f"GPT text: {args.gpt_text}",
        f"Candidates: {args.cands}",
        f"Observed JS(personal || rewritten) = {obs:.6f}",
        f"Bootstrap runs = {args.runs}, seed = {args.seed}",
        f"Bootstrap JS mean = {sum(bootstrap_js)/len(bootstrap_js):.6f}",
        f"Bootstrap JS min/max = {min(bootstrap_js):.6f}/{max(bootstrap_js):.6f}",
        f"p-value (JS <= observed) = {p_value:.4f}",
    ]
    out_path = Path(args.out).resolve()
    out_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\n[SAVED] Summary written to {out_path}")


if __name__ == "__main__":
    main()
