'''
IMPORT NECESSARY PACKAGES AND LIBRARIES
'''

import numpy as np
import pandas as pd 
import scipy
import sklearn as sk
import tqdm
import regex
import pydantic
import re 
import os 
import matplotlib as plt
import seaborn as sns 
import nltk
import spacy
import benepar
import typing 
import unicodedata
import json
import pathlib
import typing
import sys
from typing import Iterator, Optional
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords   
from collections import Counter
from nltk import sent_tokenize
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *



nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
benepar.download('benepar_en3')

### benepar_en3: Berkeley Neural Parser English model, v3     ###
### using this because i don't have enough personal writing   ###
### samples to train my own parser                            ###


# load benepar model 
def load_benepar_model(model_name: str = 'benepar_en3') -> benepar.Parser:
    """
    returns: parser
    accepts: token lists 
    outputs: NLTK trees
    """
    try:
        parser = benepar.Parser(model_name)
    except ValueError as e:
        print(
            f"Error loading benepar model '{model_name}'. "
            f"Make sure you've downloaded it with:\n"
            f"  python -m benepar.download '{model_name}'",
            file=sys.stderr,
        )
        raise e
    return parser

def parse_sentence(parser: benepar.Parser, sent_text: str) -> Optional[dict]:
    """
    Tokenize and parse a single sentence string using benepar parser. 
    Returns dict with:
        tokens: list of tokens
        tree_str: bracketed tree string 
    or None if parsing fails. 
    """
    
    #basic tokenization
    tokens = word_tokenize(sent_text)
    
    #Benepar expects tokenized input as list of strings
    try:
        tree = parser.parse(tokens)
    except Exception as e:
        #if parsing fails, skip sentence
        print(f"[WARN] Parse failed for sentence: {sent_text!r}\n  Error: {e}", file=sys.stderr)
        return None
    
    #Penn-Treebank style bracketing for tree
    tree_str = tree.pformat(margin=1_000_000) #single line tree
    
    return {
        "tokens": tokens,
        "tree": tree_str
    }

def process_clean_file(in_path: str = "gpt_data/gpt_clean.jsonl", 
                       out_path: str = "gpt_data/gpt_trees.jsonl",
                       max_sentences: Optional[int] = None) -> None:
    """
    1. Read sentences from mine_clean.jsonl
    2. parse each sentence into constitiency tree using benepar parser
    3. write a JSONL file where each line has:
        - doc_id, para_id, sent_id, text
        - tokens: list[str]
        - tree: bracketed tree string (constituency parse)
    """
    parser = load_benepar_model("benepar_en3")
    
    in_file = pathlib.Path(in_path)
    out_file = pathlib.Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    num_in = 0
    num_parsed = 0
    
    with in_file.open("r", encoding="utf-8") as fin, \
    out_file.open("w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue

            num_in += 1 
            if max_sentences is not None and num_in > max_sentences:
                break
            
            try: 
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping JSON line: {line!r}", file=sys.stderr)
                continue
            
            text = record.get("text", "")
            if not text:
                continue 
                
            parsed = parse_sentence(parser, text)
            if parsed is None: 
                #skip unparsable sentences but keep count 
                continue
            
            num_parsed += 1
            out_record = {
                "doc_id": record.get("doc_id"), 
                "para_id": record.get("para_id"),
                "sent_id": record.get("sent_id"),  
                "text": text,
                "tokens": parsed["tokens"],
                "tree": parsed["tree"],
            }
            
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            
        print(f"Read {num_in} sentences from {in_path}")
        print(f"Successfully parsed {num_parsed} sentences to {out_path}")
        
def main():
    
    process_clean_file(
        in_path = "gpt_data/gpt_clean.jsonl",
        out_path = "gpt_data/gpt_trees.jsonl",
        max_sentences=None)

if __name__ == "__main__":
    main() 