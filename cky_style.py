import json
import math
import pathlib
from typing import Dict, List, Tuple, Any, Optional

from nltk import Tree, Nonterminal
from nltk.tokenize import word_tokenize

LOG_ZERO = -1e9  #approx log(0)


class PCFG:
    """
    Loads learned PCFG from JSON, exposes binary + lexical rule tables
    for CKY, rule-lvl scoring
    """
    def __init__(self,
                 start_symbol: str = "S",
                 grammar_path: str = "models/pcfg_chatgpt.json"):
        self.start_symbol = start_symbol
        self.binary_rules: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        self.lexical_rules: Dict[str, List[Tuple[str, float]]] = {}
        self.nonterminals: set[str] = set()
        self._load(grammar_path)

    def _load(self, path: str) -> None:
        p = pathlib.Path(path)
        with p.open("r", encoding="utf-8") as f:
            pcfg = json.load(f)
        
        self.pcfg_raw = pcfg  #save for later analysis

        for lhs, rules in pcfg.items():
            self.nonterminals.add(lhs)
            for r in rules:
                rhs = r["rhs"]
                logp = r.get("log_prob", math.log(r["prob"]))
                if len(rhs) == 1:
                    #lexical rule: A -> word
                    word = rhs[0]
                    self.lexical_rules.setdefault(word, []).append((lhs, logp))
                elif len(rhs) == 2:
                    #binary rule: A -> B C
                    key = (rhs[0], rhs[1])
                    self.binary_rules.setdefault(key, []).append((lhs, logp))
                else:
                    #should not happen in good CNF
                    continue


def cky_viterbi_with_rules(tokens: List[str], grammar: PCFG) -> Tuple[Optional[float], Optional[Tree], List[float]]:
    """
    CKY with Viterbi backpointers
    Returns:
    best_logprob: float or None
    best_tree:    nltk.Tree or None
    rule_logps:   list of log-probs for each prod in best tree
    (empty list if no parse)
    """
    n = len(tokens)
    if n == 0:
        return None, None, []

    #chart: pi[(i, j)] = {A: best logprob for A covering tokens[i:j]}
    pi: Dict[Tuple[int, int], Dict[str, float]] = {}

    #backpointers: bp[(i, j, A)] = (k, B, C)
    bp: Dict[Tuple[int, int, str], Tuple[int, str, str]] = {}

    #init diagl with lexical rules
    for i, w in enumerate(tokens):
        cell: Dict[str, float] = {}

        if w in grammar.lexical_rules:
            for A, logp in grammar.lexical_rules[w]:
                cell[A] = max(cell.get(A, LOG_ZERO), logp)

        if not cell and "UNK" in grammar.lexical_rules:
            for A, logp in grammar.lexical_rules["UNK"]:
                cell[A] = max(cell.get(A, LOG_ZERO), logp)

        if not cell:
            #this is fine - some sent could have OOV words
            print(f"[DEBUG] No lexical entries for token {i}: {w!r}")

        pi[(i, i + 1)] = cell

    #dynamic prog for longer spans
    for span in range(2, n + 1):      # span length
        for i in range(0, n - span + 1):
            j = i + span
            best_for_span: Dict[str, float] = {}

            #try all split points
            for k in range(i + 1, j):
                left_cell = pi.get((i, k), {})
                right_cell = pi.get((k, j), {})
                if not left_cell or not right_cell:
                    continue

                for B, logpB in left_cell.items():
                    for C, logpC in right_cell.items():
                        key = (B, C)
                        if key not in grammar.binary_rules:
                            continue
                        for A, logp_rule in grammar.binary_rules[key]:
                            cand = logp_rule + logpB + logpC
                            if cand > best_for_span.get(A, LOG_ZERO):
                                best_for_span[A] = cand
                                bp[(i, j, A)] = (k, B, C)

            pi[(i, j)] = best_for_span

    #get best root symbol for full span
    root_cell = pi.get((0, n), {})
    if not root_cell:
        return None, None, []

    if grammar.start_symbol in root_cell:
        root_sym = grammar.start_symbol
    elif "ROOT" in root_cell:
        root_sym = "ROOT"
    else:
        #backup - best NT
        root_sym = max(root_cell.items(), key=lambda kv: kv[1])[0]

    best_logprob = root_cell[root_sym]

    #reconstruct best tree 
    def build_tree(i: int, j: int, A: str) -> Tree:
        if j == i + 1:
            return Tree(A, [tokens[i]])
        k, B, C = bp[(i, j, A)]
        left = build_tree(i, k, B)
        right = build_tree(k, j, C)
        return Tree(A, [left, right])

    best_tree = build_tree(0, n, root_sym)

    #extract rule log-probs from best tree 
    rule_logps: List[float] = []

    for prod in best_tree.productions():
        lhs = str(prod.lhs())
        rhs_syms = prod.rhs()

        #lexical rule A -> word
        if len(rhs_syms) == 1 and not isinstance(rhs_syms[0], Nonterminal):
            word = rhs_syms[0]
            # exact word
            if word in grammar.lexical_rules:
                for A, logp in grammar.lexical_rules[word]:
                    if A == lhs:
                        rule_logps.append(logp)
                        break
            #UNK backup
            elif "UNK" in grammar.lexical_rules:
                for A, logp in grammar.lexical_rules["UNK"]:
                    if A == lhs:
                        rule_logps.append(logp)
                        break

        #binary rule A -> B C
        elif len(rhs_syms) == 2 and all(isinstance(s, Nonterminal) for s in rhs_syms):
            B = str(rhs_syms[0])
            C = str(rhs_syms[1])
            key = (B, C)
            if key in grammar.binary_rules:
                for A, logp in grammar.binary_rules[key]:
                    if A == lhs:
                        rule_logps.append(logp)
                        break

    return best_logprob, best_tree, rule_logps


def compute_style_score(rule_logps: List[float]) -> Optional[float]:
    """
    Style score = average log-probability / rule.
    Higher (less negative) means more in line with learned style.
    """
    if not rule_logps:
        return None
    return sum(rule_logps) / len(rule_logps)


def parse_and_style_score(sent: str, grammar: PCFG) -> Tuple[Optional[float], Optional[Tree], Optional[float]]:
    """
    tokenize, CKY-parse, compute style score.
    Returns:
        best_logprob, best_tree, style_score
    """
    tokens = word_tokenize(sent)
    logp, tree, rule_logps = cky_viterbi_with_rules(tokens, grammar)
    if logp is None or tree is None:
        return None, None, None
    style_score = compute_style_score(rule_logps)
    return logp, tree, style_score

def rule_style_gaps(tree: Tree, grammar: PCFG) -> List[Tuple[Tree, float]]:
    """
    for each node in tree, get 'style gap' for production
       style_gap = best_logp_for_LHS - logp_for_this_rule
    returns list of (subtree, gap) for internal nodes
    """
    gaps: List[Tuple[Tree, float]] = []

    #best log-prob / LHS from grammar
    best_logp_for_lhs: Dict[str, float] = {}
    for lhs, rules in grammar.pcfg_raw.items():
        best_logp_for_lhs[lhs] = max(r["log_prob"] for r in rules)

    for node in tree.subtrees():
        if isinstance(node, Tree) and len(node) > 0:
            lhs = node.label()
            #build RHS sequence of child labels/words to match PCFG
            rhs = []
            for child in node:
                if isinstance(child, Tree):
                    rhs.append(child.label())
                else:
                    rhs.append(child)  # word

            #find logp of rule
            logp_rule = None
            for r in grammar.pcfg_raw.get(lhs, []):
                if r["rhs"] == rhs:
                    logp_rule = r["log_prob"]
                    break

            if logp_rule is None:
                #rule might not appear in collapsed PCFG
                #skip it
                continue

            best = best_logp_for_lhs.get(lhs, logp_rule)
            gap = best - logp_rule
            gaps.append((node, gap))

    return gaps


def main():
    g = PCFG(start_symbol="S", grammar_path="models/pcfg_chatgpt.json")

    sent = "Alcohol is woven into social culture so tightly that it can be hard to imagine a party, a celebration, or even a casual dinner without it."
    
    #debug
    tokens = word_tokenize(sent)
    print("TOKENS:", tokens)

    for w in tokens:
        print(w, "→", "OK" if w in g.lexical_rules else "MISSING")
        
    logp, tree, style = parse_and_style_score(sent, g)
    
    #find areas for improvement in given sentence
    gaps = rule_style_gaps(tree, g)
    #sort by gap desc - biggest gap = most off-style writing
    gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)
    
    for subtree, gap in gaps_sorted[:5]:
        print("LHS:", subtree.label(), "gap:", gap)
        print("Yield:", " ".join(subtree.leaves()))
        print()

    print("Sentence:", sent)
    print("Log-prob:", logp)
    print("Style score (avg rule log-prob):", style)
    if tree is not None:
        tree.pretty_print()


if __name__ == "__main__":
    main()