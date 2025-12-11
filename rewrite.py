from typing import List, Tuple, Optional
import argparse
import pathlib
import json

from nltk.tokenize import word_tokenize, sent_tokenize

from cky_style import PCFG, parse_and_style_score

SUPER_NEG = -1e6  #backup style score for unparsable sentences

def load_baselines(path: str = "models/style_baselines.json"):
    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["mu_you_on_you"], data["mu_gpt_on_gpt"]

#Cand gen. for sent

def generate_candidates(sent: str) -> List[str]:
    """
    cand. generator for single sent - backup 
    hard codes some 'GPT-ish -> you-ish' preferences based on style diffs:
    GPT-ish discourse markers -> simpler forms
    WHPP / QP-ish formal phrases -> simpler alternatives
    rhetorical/question-like variant for 'This ...' sents
    """
    cands = [sent]

    #discourse marker + formal opens
    replacements = [
        ("In conclusion,", "Overall,"),
        ("In conclusion ,", "Overall ,"),
        ("In conclusion ", "Overall "),
        ("in conclusion,", "overall,"),
        ("in conclusion ,", "overall ,"),
        ("However,", "But"),
        ("However ,", "But"),
        ("however,", "but"),
        ("however ,", "but"),
        ("Moreover,", "Also"),
        ("Moreover ,", "Also"),
        ("moreover,", "also"),
        ("moreover ,", "also"),
        ("Therefore,", "So"),
        ("Therefore ,", "So"),
        ("therefore,", "so"),
        ("therefore ,", "so"),
    ]

    #WHPP / QP-ish patterns (from compare_styles)
    replacements += [
        ("to which", "that"),
        ("to whom", "who"),
        ("to what extent", "how"),
        ("in many cases", "often"),
        ("in many ways", "often"),
        ("in a variety of ways", "in different ways"),
        ("in a number of ways", "in different ways"),
    ]

    for old, new in replacements:
        if old in sent:
            cands.append(sent.replace(old, new))

    #Add rhetorical/question-like piece  -  'This ...' sents
    stripped = sent.lstrip()
    if stripped.startswith("This ") or stripped.startswith("this "):
        #turn "This pattern suggests that X"
        #into "What this pattern suggests is that X"
        words = stripped.split()
        if len(words) >= 3:
            noun = words[1]
            #everything after noun
            tail = " ".join(words[2:])
            rhetorical = f"What this {noun} suggests is that {tail}"
            cands.append(rhetorical)

    #deduplicate while pres. order
    seen = set()
    deduped = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped


#rel. style scoring (you vs GPT)

def style_score_sentence_relative(
    sent: str,
    g_you: PCFG,
    g_gpt: PCFG,
    mu_you_on_you: float,
    mu_gpt_on_gpt: float,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    return normalized rel. style score for sent:
    style_you_norm = style_you - mu_you_on_you
    style_gpt_norm = style_gpt - mu_gpt_on_gpt
    style_rel = style_you_norm - style_gpt_norm

    Positive  -> closer to personal training dist than GPTs
    Negative  -> closer to GPT's training dist than personal

    return (style_you_norm, style_gpt_norm) for debug
    """
    #parse_and_style_score: (logp, tree, avg_logP_per_rule)
    _, _, style_you = parse_and_style_score(sent, g_you)
    _, _, style_gpt = parse_and_style_score(sent, g_gpt)

    if style_you is None or style_gpt is None:
        return SUPER_NEG, None, None

    style_you_norm = style_you - mu_you_on_you
    style_gpt_norm = style_gpt - mu_gpt_on_gpt
    style_rel = style_you_norm - style_gpt_norm

    return style_rel, style_you_norm, style_gpt_norm


def score_essay_relative(
    text: str,
    g_you: PCFG,
    g_gpt: PCFG,
    mu_you_on_you: float,
    mu_gpt_on_gpt: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    get overall norm. rel. style for essay
    avg_rel = mean over sents (style_you_norm - style_gpt_norm)

    return
    avg_you_norm = avg norm. style_you across sents
    avg_gpt_norm = avg norm. style_gpt across sents

    return (None, None, None) if unparsable
    """
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sents:
        return None, None, None

    rel_scores: List[float] = []
    you_scores: List[float] = []
    gpt_scores: List[float] = []

    for s in sents:
        rel, sy_norm, sg_norm = style_score_sentence_relative(
            s, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
        )
        if rel == SUPER_NEG or sy_norm is None or sg_norm is None:
            continue
        rel_scores.append(rel)
        you_scores.append(sy_norm)
        gpt_scores.append(sg_norm)

    if not rel_scores:
        return None, None, None

    avg_rel = sum(rel_scores) / len(rel_scores)
    avg_you = sum(you_scores) / len(you_scores)
    avg_gpt = sum(gpt_scores) / len(gpt_scores)

    return avg_rel, avg_you, avg_gpt


#sent-lvl rewrite using rel.  score

def choose_best_rewrite_for_sentence(
    sent: str,
    g_you: PCFG,
    g_gpt: PCFG,
    mu_you_on_you: float,
    mu_gpt_on_gpt: float,
) -> str:
    """
    input: sent + both grammars
    gen. cand. rewrites
    compute norm. rel. style scores
    select best

    objective: maximize (style_you_norm - style_gpt_norm).
    """
    candidates = generate_candidates(sent)
    scored: List[Tuple[str, float]] = []

    for cand in candidates:
        rel, sy_norm, sg_norm = style_score_sentence_relative(
            cand, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
        )
        scored.append((cand, rel))

    best_sent, best_rel = max(scored, key=lambda x: x[1])
    return best_sent


#essay-lvl rewriting

def split_into_sentences(text: str) -> List[str]:
    """
    split essay into sents - NLTK sent_tokenize
    """
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def rewrite_essay_to_your_style(
    gpt_essay: str,
    g_you: PCFG,
    g_gpt: PCFG,
    mu_you_on_you: float,
    mu_gpt_on_gpt: float,
) -> str:
    """
    rewrite GPT essay using norm. rel. style

    For each sent:
        generate cands
        score by norm. rel. style (you_norm - gpt_norm)
        pick best
    """
    sents = split_into_sentences(gpt_essay)
    rewritten_sents: List[str] = []
    changed = 0

    for s in sents:
        best = choose_best_rewrite_for_sentence(
            s, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
        )
        if best != s:
            changed += 1
        rewritten_sents.append(best)

    print(f"[INFO] Rewrote {changed} out of {len(sents)} sentences.")
    return " ".join(rewritten_sents)


#file I/O + CLI 

def read_text_file(path: str) -> str:
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Score, rewrite GPT essay toward personal style (relative to GPT's)"
    )
    parser.add_argument(
        "--mine",
        type=str,
        required=True,
        help="Path to txt file with personal essay",
    )
    parser.add_argument(
        "--gpt",
        type=str,
        required=True,
        help="Path to txt file with GPT essay to be rewritten",
    )

    args = parser.parse_args()

    #load both grammars
    g_you = PCFG(start_symbol="S", grammar_path="models/pcfg_personal.json")
    g_gpt = PCFG(start_symbol="S", grammar_path="models/pcfg_chatgpt.json")

    #load norm. baselines
    mu_you_on_you, mu_gpt_on_gpt = load_baselines("models/style_baselines.json")
    
    #read essays
    my_essay = read_text_file(args.mine)
    gpt_essay = read_text_file(args.gpt)

    #score personal essay (relative, normalized)
    my_rel, my_you_norm, my_gpt_norm = score_essay_relative(
        my_essay, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
    )
    print("=== YOUR ESSAY ===")
    print(f"Length (chars): {len(my_essay)}")
    print("Average normalized relative style (you - GPT):", my_rel)
    print("Average normalized logP_you per rule :", my_you_norm)
    print("Average normalized logP_gpt per rule :", my_gpt_norm)
    print("Interpretation: positive = closer to your training dist than GPTs\n")

    #score original GPT essay
    gpt_rel_before, gpt_you_norm_before, gpt_gpt_norm_before = score_essay_relative(
        gpt_essay, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
    )
    print("=== ORIGINAL GPT ESSAY ===")
    print(f"Length (chars): {len(gpt_essay)}")
    print("Average normalized relative style (you - GPT):", gpt_rel_before)
    print("Average normalized logP_you per rule :", gpt_you_norm_before)
    print("Average normalized logP_gpt per rule :", gpt_gpt_norm_before)
    print()

    #rewrite GPT essay
    rewritten = rewrite_essay_to_your_style(
        gpt_essay, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
    )

    #rescore rewritten essay
    gpt_rel_after, gpt_you_norm_after, gpt_gpt_norm_after = score_essay_relative(
        rewritten, g_you, g_gpt, mu_you_on_you, mu_gpt_on_gpt
    )
    print("=== REWRITTEN GPT ESSAY (STYLE-ADAPTED) ===")
    print(f"Length (chars): {len(rewritten)}")
    print("Average normalized relative style (you - GPT):", gpt_rel_after)
    print("Average normalized logP_you per rule :", gpt_you_norm_after)
    print("Average normalized logP_gpt per rule :", gpt_gpt_norm_after)

    #compare rel. style shift
    if gpt_rel_before is not None and gpt_rel_after is not None:
        delta = gpt_rel_after - gpt_rel_before
        print("=== RELATIVE STYLE SHIFT (toward you, away from GPT) ===")
        print(f"Before (you - GPT): {gpt_rel_before}")
        print(f"After  (you - GPT): {gpt_rel_after}")
        print(f"Δ (after - before): {delta:.4f}")
        if delta > 0:
            print("-> Rewritten GPT essay is more you-ish relative to GPT")
        elif delta < 0:
            print("-> Rewriting accidentally moved it toward GPT's style")
        else:
            print("-> No net change in relative style score")
    else:
        print("[INFO] Could not compute comparable essay scores")

    #save rewritten essay
    out_path = pathlib.Path("rewritten_gpt_essay_relative.txt")
    out_path.write_text(rewritten, encoding="utf-8")
    print(f"\nRewritten essay saved to: {out_path}")


if __name__ == "__main__":
    main()