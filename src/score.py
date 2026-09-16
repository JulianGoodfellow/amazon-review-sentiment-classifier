"""THE SCORING SCRIPT (assignment Steps 2 & 6).

Runs a batch of reviews through the LLM classification prompt, checks the
predictions against a rating-derived label (the model never sees the rating),
and saves per-review predictions plus aggregate metrics to results/*.json.

Two presets:
  --mode first100   Step 2: first 100 rows in file order, 2-class, no emotion.
  --mode balanced   Step 6: ~50/class balanced sample (fixed seed), 3-class,
                     with LLM primary-emotion detection.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from data import balanced_sample, label_2class, label_3class, load_reviews
from prompt import MODEL, classify_review

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def score_reviews(df, n_classes: int, with_emotion: bool, model: str) -> list[dict]:
    predictions = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        result = classify_review(
            row["title"], row["text"], n_classes=n_classes,
            with_emotion=with_emotion, model=model,
        )
        true_label = label_2class(row["rating"]) if n_classes == 2 else label_3class(row["rating"])
        predictions.append({
            "review_id": int(row["review_id"]),
            "title": row["title"],
            "text": row["text"],
            "rating": row["rating"],
            "verified_purchase": bool(row.get("verified_purchase")),
            "true_label": true_label,
            "predicted_label": result["sentiment"],
            "rationale": result["rationale"],
            "llm_primary_emotion": result.get("primary_emotion"),
        })
        if i % 20 == 0 or i == total:
            print(f"  scored {i}/{total}")
    return predictions


def compute_metrics(predictions: list[dict], labels: list[str]) -> dict:
    n = len(predictions)
    correct = sum(1 for p in predictions if p["predicted_label"] == p["true_label"])
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for p in predictions:
        confusion[p["true_label"]][p["predicted_label"]] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_as = sum(confusion[t][label] for t in labels)
        precision = tp / predicted_as if predicted_as else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "support": support, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
        }

    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def run(mode: str, model: str = MODEL, n_per_class: int = 50, seed: int = 42, with_emotion: bool | None = None) -> Path:
    df = load_reviews()

    if mode == "first100":
        n_classes = 2
        if with_emotion is None:
            with_emotion = False
        subset = df.head(100).copy()
        labels = ["POSITIVE", "NEGATIVE"]
        out_path = RESULTS_DIR / "run_2class_first100.json"
    elif mode == "balanced":
        n_classes = 3
        if with_emotion is None:
            with_emotion = True
        df["true_label_3class"] = df["rating"].apply(label_3class)
        subset = balanced_sample(df, label_col="true_label_3class", n_per_class=n_per_class, seed=seed)
        labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        out_path = RESULTS_DIR / "run_3class_balanced150.json"
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(f"Scoring {len(subset)} reviews (mode={mode}, n_classes={n_classes}, model={model})...")
    predictions = score_reviews(subset, n_classes, with_emotion, model)
    metrics = compute_metrics(predictions, labels)

    payload = {
        "mode": mode,
        "n_classes": n_classes,
        "with_emotion": with_emotion,
        "model": model,
        "seed": seed if mode == "balanced" else None,
        "n_per_class": n_per_class if mode == "balanced" else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "metrics": metrics,
        "predictions": predictions,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved -> {out_path}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    return out_path


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["first100", "balanced"], required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emotion", dest="with_emotion", action="store_true", default=None)
    parser.add_argument("--no-emotion", dest="with_emotion", action="store_false")
    args = parser.parse_args()
    run(args.mode, model=args.model, n_per_class=args.n_per_class, seed=args.seed, with_emotion=args.with_emotion)
