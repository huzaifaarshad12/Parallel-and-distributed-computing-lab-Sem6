#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# download_agnews.py  --  the ONE real dataset for this project (AG News)
# ---------------------------------------------------------------------------
# Downloads the AG News dataset from Hugging Face and writes the articles as
# plain text files (one article per line) that the C++ pipeline can read.
#
# AG News is a real news-classification dataset of ~120,000 training articles in
# 4 categories (World, Sports, Business, Sci/Tech). Each example has a short
# article text. We discard the labels (preprocessing only needs the raw text)
# and emit one article per line.
#
# To keep the project usable on a normal laptop AND to give us several dataset
# SIZES for benchmarking the parallel versions (Phases 2 & 3), this script
# writes four files that are all REAL AG News text:
#
#       data/agnews/agnews_1k.txt     first   1,000 articles
#       data/agnews/agnews_10k.txt    first  10,000 articles
#       data/agnews/agnews_50k.txt    first  50,000 articles
#       data/agnews/agnews_full.txt   all (~120,000) articles
#
# The smaller files are exact prefixes of the larger ones, so timings across
# sizes are directly comparable.
#
# Requirements (install once):
#   pip install datasets
#
# Usage:
#   python scripts/download_agnews.py                 # writes all four files
#   python scripts/download_agnews.py 20000           # also writes a custom size
# ---------------------------------------------------------------------------

import os
import sys

OUT_DIR = os.path.join("data", "agnews")

# (filename, number_of_articles).  None means "all available".
DEFAULT_SIZES = [
    ("agnews_1k.txt", 1_000),
    ("agnews_10k.txt", 10_000),
    ("agnews_50k.txt", 50_000),
    ("agnews_full.txt", None),
]


def clean_line(text):
    """Collapse any internal newlines/tabs so each article stays on one line.
    (We do NOT lowercase or strip punctuation here -- that is the C++ pipeline's
    job. This only guarantees the line-based file format.)"""
    return " ".join(text.split())


def write_subset(articles, count, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(articles) if count is None else min(count, len(articles))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            f.write(articles[i] + "\n")
    print(f"  wrote {n:>6} articles -> {path}")


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("The 'datasets' package is required. Install it with:")
        print("    pip install datasets")
        sys.exit(1)

    print("Downloading AG News from Hugging Face (cached locally after first run)...")
    # The canonical repo is "fancyzhx/ag_news"; the bare "ag_news" alias is no
    # longer resolvable on newer `datasets` versions, so we try them in order.
    ds = None
    last_err = None
    for repo_id in ("fancyzhx/ag_news", "ag_news"):
        try:
            ds = load_dataset(repo_id, split="train")
            print(f"  (loaded from '{repo_id}')")
            break
        except Exception as e:  # noqa: BLE001 - report and try the next id
            last_err = e
            print(f"  could not load '{repo_id}': {e}")
    if ds is None:
        raise SystemExit(f"Failed to load AG News. Last error:\n{last_err}")

    # Materialize the cleaned article texts once, then slice into sizes.
    articles = []
    for ex in ds:
        text = clean_line(ex.get("text", ""))
        if text:
            articles.append(text)
    print(f"Loaded {len(articles)} real articles.\n")

    sizes = list(DEFAULT_SIZES)
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        sizes.append((f"agnews_{n}.txt", n))

    for name, count in sizes:
        write_subset(articles, count, os.path.join(OUT_DIR, name))

    print("\nDone. Now build and run, e.g.:")
    print("    .\\build.ps1")
    print("    build\\seq_pipeline.exe data\\agnews\\agnews_10k.txt results\\clean_10k.txt 5")


if __name__ == "__main__":
    main()
