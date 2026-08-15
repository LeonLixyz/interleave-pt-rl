# scripts/data/make_dataset.py
import os, sys, argparse, pathlib

# --- ensure project root on sys.path (no install needed) ---
repo_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# --- imports from your package ---
from llm_tokens.chess import parquet_to_raw_txt, parquet_to_txt_shards, SanTokenizer, UciTokenizer
from llm_tokens.chess import PgnTokenizer as LibPGNTokenizer
from llm_tokens.chess import BpeTokenizer
from llm_tokens.chess.bpe_tokenizer import train_bpe 

if __name__ == "__main__":
    tokenizer = UciTokenizer()
    ids = tokenizer.encode("1. a4 h5 2. a5 h4 3. a6 h3 4. axb7 hxg2 5. bxa8=Q *")
    print(ids)
    print(tokenizer.decode(ids))