import json
import math
import pathlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import re
import regex

import nltk
from nltk import Tree, Nonterminal, Production

#strip func to rm func tags from NT's
def strip_functional(tag: str) -> str:
    return tag.split("-")[0].split("=")[0]

def load_trees(jsonl_path: str) -> List[Tree]:
    """
    load constituency trees from JSONL file from pipeline
    each line should be JSON obj with "tree" with bracketed tree str
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
    Convert tree to CNF, return lst of prods
    CNF req for CKY parsing - all prods A -> BC or A -> a
    """
    
    #work on copy of tree so don't modify / mess up original
    t = t.copy(deep=True)
    
    for subtree in t.subtrees():
        if isinstance(subtree.label(), str):
            subtree.set_label(strip_functional(subtree.label()))
    
    #remove unary nodes
    t.collapse_unary(collapsePOS=False, collapseRoot=False)

    #collapse branches 
    #factor = 'left' controls how to branch
    t.chomsky_normal_form(factor='left', horzMarkov=2)
    
    #after CNF conversion, extract prods (NT) and lexical)
    prods = t.productions()
    return list(prods)

def collect_productions(trees: List[Tree]) -> List[Production]:
    """
    Collect CNF prods from trees
    """
    all_prods: List[Production] = []
    for i, t in enumerate(trees, start=1):
        prods = tree_to_cnf_productions(t)
        all_prods.extend(prods)
        if i % 100 == 0:
            print(f"Processed {i} trees... total productions so far: {len(all_prods)}")
    print(f"Collected total of {len(all_prods)} productions from {len(trees)} trees")
    return all_prods

def estimate_pcfg(prods: List[Production], 
                  min_count: int = 1, 
                  top_k: int = 9999) -> Dict[str, List[Dict[str, Any]]]:
    """
    est. PCFG rule probs from list of prods
    P(A -> beta) = count(A -> beta) / sum_{all beta'} count(A -> beta')
    
    return dict mapping LHS NT str to list of rule - each rule:
    {"rhs": [sym1, sym2, ...], "prob": float, "log_prob": float}
    
    args:
    min_count: drop rules where count < min_count
    top_k: keep max top_k rules per LHS by prob
    """
    
    
    # counts[lhs][rhs_tuple] = freq
    counts: Dict[Nonterminal, Counter] = defaultdict(Counter)
    
    for p in prods:
        lhs = p.lhs()
        rhs = p.rhs()
        counts[lhs][rhs] += 1 
        
    pcfg: Dict[str, List[Dict[str, Any]]] = {}
    
    for lhs, rhs_counter in counts.items():
        #remove low freq rules
        filtered = {rhs: c for rhs, c in rhs_counter.items() if c >= min_count}
        if not filtered:
            continue
    
        total = sum(filtered.values())
        lhs_str = str(lhs)
        
        # keep reg phrase/POS tags (A–Z, $) or known punct tags
        # allowed_punct = {".", ",", ":", "''", "``", "-LRB-", "-RRB-"}
        # if not (re.match(r"^[A-Z$]+$", lhs_str) or lhs_str in allowed_punct):
        #drop synthetic stuff like 'NP|<NP-PP>'
        #cont
        
        #compute probs for each RHS
        rule_list: List[Tuple[Tuple[Any, ...], float]] = []
        for rhs, c in filtered.items():
            prob = c / total
            rule_list.append((rhs, prob))
            
        #sort by desc prob, keep top_k
        rule_list.sort(key=lambda x: x[1], reverse=True)
        if len(rule_list) > top_k:
            rule_list = rule_list[:top_k]
        
        #RHS symbols to strings for JSON
        serialized_rules: List[Dict[str, Any]] = []
        for rhs, prob in rule_list:
            rhs_syms: List[str]= []
            for sym in rhs:
                if isinstance(sym, Nonterminal):
                    rhs_syms.append(str(sym))
                else:
                    #terminal symbol (word)
                    rhs_syms.append(sym)
            serialized_rules.append(
                {
                        "rhs": rhs_syms, 
                        "prob": prob, 
                        "log_prob": math.log(prob)
                }
            )
            
        pcfg[lhs_str] = serialized_rules
        
    print(f"Built PCFG with {len(pcfg)} LHS nonterminals")
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
    trees = load_trees("data/mine_trees.jsonl")
    prods = collect_productions(trees)
    
    pcfg = estimate_pcfg(prods, min_count=1, top_k=9999)
    save_pcfg(pcfg, "models/pcfg_personal.json")

if __name__ == "__main__":
    main()
        