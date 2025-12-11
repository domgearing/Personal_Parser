import argparse
import json
import math
import pathlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import re
import regex

import nltk
from nltk import Tree, Nonterminal, Production

#strip func to remove functional tags from NT's
def strip_functional(tag: str) -> str:
    return tag.split("-")[0].split("=")[0]

#collapse to base lab to reduce # of NTs
def base_label(sym: str) -> str:
    """
    Collapse function annotated labels to base cat.
    Ex 'NP|<DT-JJ>' -> 'NP'
    """
    s = sym
    #rm anything after '|'
    if "|" in s:
        s = s.split("|")[0]
    #rm function tags after '-' or '=' 
    if "-" in s:
        s = s.split("-")[0]
    if "=" in s:
        s = s.split("=")[0]
    return s

def load_trees(jsonl_path: str) -> List[Tree]:
    """
    load constituency trees from JSONL file from pipeline 
    each line should be JSON object with "tree" with bracketed tree string
    return list of nltk.Tree obj
    """
    
    trees: List[Tree] = []
    
    path = pathlib.Path(jsonl_path)
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tree_str = rec.get("tree")
            if not tree_str:
                continue
            try:
                t = Tree.fromstring(tree_str)
            except Exception as e:
                print(f"[WARN] Could not parse tree string: {e}")
                continue
            trees.append(t)
    
    print(f"Loaded {len(trees)} trees from {jsonl_path}")
    
    return trees

def tree_to_cnf_productions(t: Tree) -> List[Production]:
    """
    convert tree to Chomsky Normal Form (CNF), return list of prods
    CNF required for CKY parsing, all prods either A -> BC or A -> a
    """
    
    #work on copy of tree so don't modify / mess up original
    t = t.copy(deep=True)
    
    for subtree in t.subtrees():
        if isinstance(subtree.label(), str):
            subtree.set_label(strip_functional(subtree.label()))
    
    #remove unary nodes
    t.collapse_unary(collapsePOS=False, collapseRoot=False)

    #collapse branches 
    #factor = 'left' controls how branching done
    t.chomsky_normal_form(factor='left', horzMarkov=1)
    
    #after CNF conversion, extract productions (NT and lexical)
    prods = t.productions()
    return list(prods)

def collect_productions(trees: List[Tree]) -> List[Production]:
    """
    collect CNF prods from all trees
    """
    all_prods: List[Production] = []
    for i, t in enumerate(trees, start=1):
        prods = tree_to_cnf_productions(t)
        all_prods.extend(prods)
        if i % 100 == 0:
            print(f"Processed {i} trees... total productions so far: {len(all_prods)}")
    print(f"Collected total of {len(all_prods)} productions from {len(trees)} trees")
    return all_prods

def estimate_pcfg(
    prods: List[Production],
    min_count: int = 1,
    top_k: int = 9999,
    #`rare_threshold` param in `estimate_pcfg` func used for rare words in PCFG
    rare_threshold: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    est. PCFG rule probs from list of CNF prods
    UNK handling for rare words
    words with count <= rare_threshold replaced by 'UNK'
    """

    #count word freqs
    word_counts: Counter = Counter()
    for p in prods:
        for sym in p.rhs():
            if not isinstance(sym, Nonterminal):
                word_counts[sym] += 1

    #collect counts with UNK 
    counts: Dict[str, Counter] = defaultdict(Counter)

    def base_label(sym: str) -> str:
        s = sym
        if "|" in s:
            s = s.split("|")[0]
        if "-" in s:
            s = s.split("-")[0]
        if "=" in s:
            s = s.split("=")[0]
        return s

    for p in prods:
        lhs_label = base_label(str(p.lhs()))

        rhs_labels: List[str] = []
        for sym in p.rhs():
            if isinstance(sym, Nonterminal):
                rhs_labels.append(base_label(str(sym)))
            else:
                word = sym
                if word_counts[word] < rare_threshold:
                    rhs_labels.append("UNK")
                else:
                    rhs_labels.append(word)

        counts[lhs_label][tuple(rhs_labels)] += 1

    #counts -> probs
    pcfg: Dict[str, List[Dict[str, Any]]] = {}

    for lhs_str, rhs_counter in counts.items():
        #drop rare rules 
        filtered = {rhs: c for rhs, c in rhs_counter.items() }#if c > min_count}
        # if not filtered:
        # print(c)
        #continue

        total = sum(filtered.values())

        rule_list: List[Tuple[Tuple[Any, ...], float]] = []
        for rhs, c in filtered.items():
            prob = c / total
            rule_list.append((rhs, prob))

        #limit rules to top_k / LHS
        rule_list.sort(key=lambda x: x[1], reverse=True)
        if len(rule_list) > top_k:
            rule_list = rule_list[:top_k]

        serialized_rules: List[Dict[str, Any]] = []
        for rhs, prob in rule_list:
            rhs_syms = list(rhs)
            serialized_rules.append(
                {
                    "rhs": rhs_syms,
                    "prob": prob,
                    "log_prob": math.log(prob),
                }
            )

        pcfg[lhs_str] = serialized_rules

    print(f"Built PCFG with {len(pcfg)} LHS nonterminals (with UNK)")
    return pcfg

def save_pcfg(pcfg: Dict[str, List[Dict[str, Any]]], out_path: str) -> None:
    """
    Save PCFG dict to JSON
    """
    path=pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(pcfg, f, ensure_ascii=False, indent=2)
    print(f"Saved PCFG with {len(pcfg)} LHS symbols to {out_path}")       
    
def main():
    parser = argparse.ArgumentParser(
        description="Est. lexical PCFG from constituency trees with UNK handling"
    )
    parser.add_argument(
        "--trees",
        type=str,
        required=True,
        help="Path to input JSONL with bracketed trees",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for PCFG JSON",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum rule count to keep (default: 1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=9999,
        help="Limit # rules per LHS (default is 9999 = no cap)",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=1,
        help="Words with count <= threshold become UNK (default: 1)",
    )
    args = parser.parse_args()

    trees = load_trees(args.trees)
    prods = collect_productions(trees)
    pcfg = estimate_pcfg(
        prods,
        min_count=args.min_count,
        top_k=args.top_k,
        rare_threshold=args.rare_threshold,
    )
    save_pcfg(pcfg, args.out)

if __name__ == "__main__":
    main()
        
