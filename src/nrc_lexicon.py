"""Fetch and cache the NRC Word-Emotion Association Lexicon (EmoLex).

Mohammad, S.M. & Turney, P.D. (2013). "Crowdsourcing a Word-Emotion
Association Lexicon." Computational Intelligence, 29(3), 436-465.
https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

Free for research use. Cached locally under data/ (gitignored, like the
review data) - not redistributed in the repo.
"""
import io
import zipfile
from pathlib import Path

import requests

NRC_URL = "https://saifmohammad.com/WebDocs/NRC-Emotion-Lexicon.zip"
NRC_INNER_PATH = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHED_TXT = DATA_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def download_if_needed(path: Path = CACHED_TXT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "*/*",
    }
    resp = requests.get(NRC_URL, timeout=60, headers=headers)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open(NRC_INNER_PATH) as src, open(path, "wb") as dst:
            dst.write(src.read())
    return path


def load_lexicon(path: Path = CACHED_TXT) -> dict[str, set[str]]:
    """Return {emotion: set(words)} for the 8 NRC emotion categories."""
    download_if_needed(path)
    lexicon: dict[str, set[str]] = {e: set() for e in EMOTIONS}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, category, assoc = parts
            if category in lexicon and assoc == "1":
                lexicon[category].add(word)
    return lexicon


if __name__ == "__main__":
    lex = load_lexicon()
    for emotion, words in lex.items():
        print(f"{emotion:>13}: {len(words):>5} words  (e.g. {sorted(words)[:5]})")
