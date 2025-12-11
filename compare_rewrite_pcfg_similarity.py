"""
compare_rewrite_pcfg_similarity.py

Combine all rewritten GPT essays from earlier runs, build POS PCFG over them
compare:  
    personal PCFG vs original GPT PCFG
    personal PCFG vs rewritten PCFG

Similarity metric: Jensen-Shannon divergence over rule distributions
(rules flattened "LHS -> RHS1 RHS2 ...").

uses existing scripts (process_sample_text.py, make_POS_trees.py, pcfg_POS.py) 
to build rewritten PCFG
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

from cky_POS import PCFG, parse_and_style_score

SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: List[str], cwd: Path):
    #invoke script with current interpreter
    if cmd and cmd[0].endswith(".py"):
        cmd = [sys.executable] + cmd
    subprocess.run(cmd, check=True, cwd=str(cwd))


def load_rule_probs(pcfg_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Load PCFG JSON, return per-LHS dist. {lhs: {rhs_str: prob}}
    """
    data = json.loads(pcfg_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, float]] = {}
    for lhs, rules in data.items():
        dist = {}
        for r in rules:
            rhs = " ".join(r["rhs"])
            dist[rhs] = r["prob"]
        out[lhs] = dist
    return out


def jensen_shannon(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-9) -> float:
    """
    Jensen-Shannon divergence between two dist. rep. as dicts
    """
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


def per_lhs_js(
    lhs_dist_p: Dict[str, Dict[str, float]],
    lhs_dist_q: Dict[str, Dict[str, float]],
    eps: float = 1e-6,
) -> float:
    """
    get wghtd avg JS across LHS dists
    Weight = average total mass of LHS dists (should be around 1 if norm.)
    """
    lhs_keys = set(lhs_dist_p) | set(lhs_dist_q)
    total_weight = 0.0
    acc = 0.0
    for lhs in lhs_keys:
        dp = lhs_dist_p.get(lhs, {})
        dq = lhs_dist_q.get(lhs, {})
        keys = set(dp) | set(dq)
        #smooth, renormalize per LHS
        p_norm = {k: (dp.get(k, 0.0) + eps) for k in keys}
        q_norm = {k: (dq.get(k, 0.0) + eps) for k in keys}
        sum_p = sum(p_norm.values())
        sum_q = sum(q_norm.values())
        p_norm = {k: v / sum_p for k, v in p_norm.items()}
        q_norm = {k: v / sum_q for k, v in q_norm.items()}
        js = jensen_shannon(p_norm, q_norm, eps=eps)
        weight = 0.5 * (sum_p + sum_q)
        acc += weight * js
        total_weight += weight
    return acc / total_weight if total_weight > 0 else float("nan")


def split_sents(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def avg_logprob_under_personal(grammar_path: Path, text_path: Path) -> Optional[float]:
    """
    Avg sent log-prob of text under personal POS PCFG
    Return None if no sent parse
    """
    g = PCFG(start_symbol="S", grammar_path=str(grammar_path))
    text = text_path.read_text(encoding="utf-8")
    sents = split_sents(text)
    logps = []
    for s in sents:
        logp, _, _, _ = parse_and_style_score(s, g)
        if logp is not None:
            logps.append(logp)
    if not logps:
        return None
    return sum(logps) / len(logps)


def build_rewrite_pcfg(rewrite_files: List[Path], out_dir: Path) -> Path:
    """
    Concat rewrites, run POS tree + PCFG est., return PCFG path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_txt = out_dir / "rewrites_concat.txt"
    concat_jsonl = out_dir / "rewrites_clean.jsonl"
    rew_trees = out_dir / "rewrites_pos_trees.jsonl"
    rew_pcfg = out_dir / "rewrites_pcfg_pos.json"

    with concat_txt.open("w", encoding="utf-8") as out:
        for f in rewrite_files:
            out.write(f.read_text(encoding="utf-8"))
            out.write("\n")

    #preprocess -> POS trees -> PCFG
    run_cmd(
        [
            str(SCRIPT_DIR / "process_sample_text.py"),
            "--in-dir",
            str(concat_txt.parent),
            "--out-path",
            str(concat_jsonl),
        ],
        cwd=SCRIPT_DIR,
    )
    run_cmd(
        [
            str(SCRIPT_DIR / "make_POS_trees.py"),
            "--in-path",
            str(concat_jsonl),
            "--out-path",
            str(rew_trees),
        ],
        cwd=SCRIPT_DIR,
    )
    run_cmd(
        [
            str(SCRIPT_DIR / "pcfg_POS.py"),
            "--trees",
            str(rew_trees),
            "--out",
            str(rew_pcfg),
        ],
        cwd=SCRIPT_DIR,
    )
    return rew_pcfg


def main():
    parser = argparse.ArgumentParser(
        description="Compare PCFG similarity: personal vs original GPT, personal vs rewritten GPT."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Root containing run folders with rewritten_gpt_essay_relative_pos.txt.",
    )
    parser.add_argument(
        "--personal-pcfg",
        type=str,
        default=None,
        help="Path to personal PCFG JSON (POS-based). If not given, take most recent in runs/*/models/pcfg_personal_pos.json",
    )
    parser.add_argument(
        "--gpt-pcfg",
        type=str,
        default=None,
        help="Path to original GPT PCFG JSON (POS-based). If not given, take most recent in runs/*/models/pcfg_gpt_pos.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="rewrite_pcfg_similarity.txt",
        help="loc to write summary text output",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_dir).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(runs_root)

    #process each run (LOOCV style)
    run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    summaries = []
    aggregated_rewrites: List[Path] = []

    for rd in run_dirs:
        personal_pcfg = rd / "models" / "pcfg_personal_pos.json"
        gpt_pcfg = rd / "models" / "pcfg_gpt_pos.json"
        rewrite_txt = rd / "rewritten_gpt_essay_relative_pos.txt"
        #pick holdout GPT clean text (skip train)
        gpt_texts = [p for p in (rd / "gpt_data").glob("*clean*.txt") if "train" not in p.name]
        gpt_holdout = gpt_texts[0] if gpt_texts else None
        if not (personal_pcfg.exists() and gpt_pcfg.exists() and rewrite_txt.exists()):
            continue
        if gpt_holdout is None:
            continue

        aggregated_rewrites.append(rewrite_txt)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"rewrite_pcfg_{rd.name}_"))
        try:
            rew_pcfg_path = build_rewrite_pcfg([rewrite_txt], tmp_dir)

            personal_lhs = load_rule_probs(personal_pcfg)
            gpt_lhs = load_rule_probs(gpt_pcfg)
            rew_lhs = load_rule_probs(rew_pcfg_path)

            js_personal_gpt = per_lhs_js(personal_lhs, gpt_lhs)
            js_personal_rew = per_lhs_js(personal_lhs, rew_lhs)

            avg_logp_orig = avg_logprob_under_personal(personal_pcfg, gpt_holdout)
            avg_logp_rew = avg_logprob_under_personal(personal_pcfg, rewrite_txt)

            summaries.append(
                {
                    "run": rd.name,
                    "js_personal_gpt": js_personal_gpt,
                    "js_personal_rew": js_personal_rew,
                    "personal_pcfg": personal_pcfg,
                    "gpt_pcfg": gpt_pcfg,
                    "avg_logp_orig": avg_logp_orig,
                    "avg_logp_rew": avg_logp_rew,
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not summaries:
        raise FileNotFoundError("No runs with req. PCFGs and rewrites found")

    lines = []
    for s in summaries:
        lines.extend(
            [
                f"Run: {s['run']}",
                f"Personal PCFG: {s['personal_pcfg']}",
                f"GPT PCFG: {s['gpt_pcfg']}",
                f"JS(personal || original_gpt) = {s['js_personal_gpt']:.6f}",
                f"JS(personal || rewritten_gpt) = {s['js_personal_rew']:.6f}",
                f"Avg logP (orig under personal) = {s['avg_logp_orig']}",
                f"Avg logP (rewritten under personal) = {s['avg_logp_rew']}",
                "",
            ]
        )
    avg_js_gpt = sum(s["js_personal_gpt"] for s in summaries) / len(summaries)
    avg_js_rew = sum(s["js_personal_rew"] for s in summaries) / len(summaries)
    avg_logp_orig = (
        sum(s["avg_logp_orig"] for s in summaries if s["avg_logp_orig"] is not None)
        / sum(1 for s in summaries if s["avg_logp_orig"] is not None)
        if summaries
        else float("nan")
    )
    avg_logp_rew = (
        sum(s["avg_logp_rew"] for s in summaries if s["avg_logp_rew"] is not None)
        / sum(1 for s in summaries if s["avg_logp_rew"] is not None)
        if summaries
        else float("nan")
    )
    lines.append(f"Average JS(personal || original_gpt) across runs: {avg_js_gpt:.6f}")
    lines.append(f"Average JS(personal || rewritten_gpt) across runs: {avg_js_rew:.6f}")
    lines.append(f"Average logP(orig under personal) across runs: {avg_logp_orig}")
    lines.append(f"Average logP(rewritten under personal) across runs: {avg_logp_rew}")
    lines.append(
        "Lower JS means closer distributions. If JS(personal||rewritten) < JS(personal||gpt), rewrites moved toward personal style"
    )

    #comb all rewrites into 1 PCFG, compare to personal PCFG from most recent run
    if aggregated_rewrites:
        tmp_dir = Path(tempfile.mkdtemp(prefix="rewrite_pcfg_all_"))
        try:
            rew_all_pcfg = build_rewrite_pcfg(aggregated_rewrites, tmp_dir)
            #pick ref personal PCFG
            personal_ref = max(
                [s["personal_pcfg"] for s in summaries], key=lambda p: p.stat().st_mtime
            )
            personal_lhs_ref = load_rule_probs(personal_ref)
            rew_all_lhs = load_rule_probs(rew_all_pcfg)
            js_all = per_lhs_js(personal_lhs_ref, rew_all_lhs)
            lines.append("")
            lines.append(f"Aggregated rewrites PCFG: {rew_all_pcfg}")
            lines.append(f"JS(personal_ref || aggregated rewrites) = {js_all:.6f}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    out_path = Path(args.out).resolve()
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[SAVED] Summary written to {out_path}")


if __name__ == "__main__":
    main()
