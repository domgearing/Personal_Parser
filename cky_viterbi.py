import json
import math
import pathlib
from typing import Dict, List, Tuple, Any, Optional

from nltk import Tree

from nltk.tokenize import word_tokenize


LOG_ZERO = -1e9  #approx log(0)


class PCFG:
    def __init__(self,
                 start_symbol: str = "S",
                 grammar_path: str = "models/pcfg_personal.json"):
        self.start_symbol = start_symbol
        self.binary_rules: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        self.lexical_rules: Dict[str, List[Tuple[str, float]]] = {}
        self.nonterminals: set[str] = set()
        self._load(grammar_path)

    def _load(self, path: str) -> None:
        p = pathlib.Path(path)
        with p.open("r", encoding="utf-8") as f:
            pcfg = json.load(f)

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


def cky_viterbi(tokens: List[str], grammar: PCFG) -> Tuple[Optional[float], Optional[Tree]]:
    """
    CKY with Viterbi backpointers.
    Returns (best_logprob, best_tree) or (None, None) if no parse
    """
    n = len(tokens)
    if n == 0:
        return None, None

    #chart: pi[(i, j)] = {A: best logprob for A covering tokens[i:j]}
    pi: Dict[Tuple[int, int], Dict[str, float]] = {}

    #backpointers: bp[(i, j, A)] = (k, B, C)
    # A(i,j) -> B(i,k) C(k,j)
    bp: Dict[Tuple[int, int, str], Tuple[int, str, str]] = {}

    #init diag with lexical rules
    for i, w in enumerate(tokens):
        cell: Dict[str, float] = {}

        if w in grammar.lexical_rules:
            for A, logp in grammar.lexical_rules[w]:
                cell[A] = max(cell.get(A, LOG_ZERO), logp)

        if not cell and "UNK" in grammar.lexical_rules:
            for A, logp in grammar.lexical_rules["UNK"]:
                cell[A] = max(cell.get(A, LOG_ZERO), logp)

        if not cell:
            print(f"[DEBUG] No lexical entries for token {i}: {w!r}")

        pi[(i, i + 1)] = cell

    #dynamic programming for longer spans
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
                                #backpointer for A spanning (i,j)
                                bp[(i, j, A)] = (k, B, C)

            pi[(i, j)] = best_for_span

    #choose best root symbol for full span
    root_cell = pi.get((0, n), {})
    if not root_cell:
        return None, None

    if grammar.start_symbol in root_cell:
        root_sym = grammar.start_symbol
    elif "ROOT" in root_cell:
        root_sym = "ROOT"
    else:
        #fallback best nonterminal
        root_sym = max(root_cell.items(), key=lambda kv: kv[1])[0]

    best_logprob = root_cell[root_sym]

    #reconstruct tree via backpointers
    def build_tree(i: int, j: int, A: str) -> Tree:
        #lexical span - just attach the word
        if j == i + 1:
            return Tree(A, [tokens[i]])

        k, B, C = bp[(i, j, A)]
        left = build_tree(i, k, B)
        right = build_tree(k, j, C)
        return Tree(A, [left, right])

    best_tree = build_tree(0, n, root_sym)
    return best_logprob, best_tree

def score_sentence(sent: str, grammar: PCFG):
    tokens = word_tokenize(sent)
    return cky_viterbi(tokens, grammar)

def main():
    
    g = PCFG(start_symbol="S", grammar_path="models/pcfg_personal.json")
    sent = "The first major shift South Africa faced came with the arrival of the Dutch and the emergence of coerced labor as a vital aspect of the developing Cape Colony's economy ."
    logp, tree = score_sentence(sent, g)
    print("Sentence:", sent)
    print("Log-prob:", logp)
    if tree is not None:
        tree.pretty_print()
        
if __name__ == "__main__":
    main()