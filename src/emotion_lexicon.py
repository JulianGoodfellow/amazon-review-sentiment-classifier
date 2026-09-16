"""THE WORD-LIST SCRIPT (assignment Step 5, method 2).

Derives each review's primary emotion from the NRC word list - no model
calls. Tokenizes the review text, sums per-emotion word hits, and takes the
argmax. Runs over an existing results/*.json (adding a lexicon_primary_emotion
field alongside the LLM's own llm_primary_emotion) so the two methods can be
compared directly.
"""
import argparse
import json
import re
from pathlib import Path

from nrc_lexicon import EMOTIONS, load_lexicon

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def score_emotions(text: str, lexicon: dict[str, set[str]]) -> dict[str, int]:
    tokens = tokenize(text)
    return {emotion: sum(1 for t in tokens if t in words) for emotion, words in lexicon.items()}


def primary_emotion(scores: dict[str, int]) -> str:
    best = max(scores.values(), default=0)
    if best == 0:
        return "none"
    # deterministic tie-break: fixed EMOTIONS order
    for emotion in EMOTIONS:
        if scores[emotion] == best:
            return emotion
    return "none"


def annotate_file(path: Path) -> None:
    lexicon = load_lexicon()
    payload = json.loads(path.read_text())
    for pred in payload["predictions"]:
        full_text = f"{pred.get('title', '')} {pred.get('text', '')}"
        scores = score_emotions(full_text, lexicon)
        pred["lexicon_emotion_scores"] = scores
        pred["lexicon_primary_emotion"] = primary_emotion(scores)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Annotated {len(payload['predictions'])} reviews in {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=Path)
    args = parser.parse_args()
    annotate_file(args.results_file)
