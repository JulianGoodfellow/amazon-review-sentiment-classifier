"""Load and sample the Amazon Reviews '23 Gift Cards category.

Source: McAuley Lab, UC San Diego — https://amazon-reviews-2023.github.io
File:   https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/Gift_Cards.jsonl.gz"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "Gift_Cards.jsonl.gz"

FIELDS = [
    "rating", "title", "text", "verified_purchase", "helpful_vote",
    "timestamp", "images", "asin", "parent_asin", "user_id",
]


def download_if_needed(url: str = DATA_URL, path: Path = RAW_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def load_reviews(path: Path = RAW_PATH) -> pd.DataFrame:
    """Stream-parse the gzipped JSONL file into a DataFrame."""
    download_if_needed(path=path)
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({k: obj.get(k) for k in FIELDS})
    df = pd.DataFrame.from_records(records)
    df["rating"] = df["rating"].astype(float)
    df["n_images"] = df["images"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df = df.drop(columns=["images"])
    df = df.reset_index(drop=True)
    df["review_id"] = df.index
    return df


def label_2class(rating: float) -> str:
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def label_3class(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def balanced_sample(
    df: pd.DataFrame,
    label_col: str = "true_label_3class",
    n_per_class: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw ~n_per_class rows per label value, fixed seed, then shuffle the result."""
    rng = np.random.default_rng(seed)
    parts = []
    for label, group in df.groupby(label_col):
        n = min(n_per_class, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        parts.append(df.loc[idx])
    sample = pd.concat(parts)
    shuffled_idx = rng.permutation(sample.index.to_numpy())
    return sample.loc[shuffled_idx].reset_index(drop=True)


if __name__ == "__main__":
    df = load_reviews()
    print(f"Loaded {len(df):,} reviews")
    print(f"Columns: {list(df.columns)}")
    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index())
    print("\nSample rows:")
    print(df[["rating", "title", "text"]].head(3).to_string())
