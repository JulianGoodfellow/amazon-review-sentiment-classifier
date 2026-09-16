"""The reusable structured classification prompt (Step 1) + emotion extension (Step 5).

Given only a review's title and text (never its star rating), asks an OpenAI
model for a clean, machine-readable verdict: sentiment class, optionally a
primary emotion, and a one-line rationale (for spot-checking, not scored).
"""
import json
import os

from openai import OpenAI

MODEL = "gpt-4o-mini"

SENTIMENT_LABELS_2 = ["POSITIVE", "NEGATIVE"]
SENTIMENT_LABELS_3 = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust", "none",
]

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _system_prompt(n_classes: int, with_emotion: bool) -> str:
    if n_classes == 2:
        class_rules = (
            "Classify the review's sentiment as exactly one of: POSITIVE, NEGATIVE.\n"
            "There is no neutral option here - if the review is mixed or lukewarm, "
            "pick whichever side it leans toward more strongly."
        )
    else:
        class_rules = (
            "Classify the review's sentiment as exactly one of: POSITIVE, NEUTRAL, NEGATIVE.\n"
            "- POSITIVE: clearly favorable, satisfied, or enthusiastic.\n"
            "- NEGATIVE: clearly unfavorable, disappointed, or frustrated.\n"
            "- NEUTRAL: lukewarm, purely factual/transactional (e.g. just confirms "
            "delivery or use), or genuinely mixed with praise and complaints that "
            "roughly balance out - not just 'somewhat positive'."
        )

    edge_cases = (
        "Edge cases to handle:\n"
        "- If the title and body seem to disagree, weight the body text more heavily - "
        "titles are short and often generic.\n"
        "- Terse reviews (\"Works.\", \"Fine.\") with no clear charge are NEUTRAL "
        "(3-class) or read from tone alone (2-class).\n"
        "- Angry or sarcastic short reviews should be read for tone, not just length.\n"
        "- You do not have access to and must not infer or reference any star rating. "
        "Base your answer only on the title and text given."
    )

    emotion_rule = ""
    if with_emotion:
        emotion_rule = (
            "\n\nAlso identify the review's single PRIMARY emotion, one of: "
            + ", ".join(EMOTIONS) + ". "
            "Use 'none' only if the text is purely factual with no discernible "
            "emotional tone. Base this on the emotional content of the writing, "
            "independent of your sentiment call."
        )

    return (
        "You are a precise review-classification assistant.\n\n"
        + class_rules + "\n\n" + edge_cases + emotion_rule
    )


def _response_schema(n_classes: int, with_emotion: bool) -> dict:
    labels = SENTIMENT_LABELS_2 if n_classes == 2 else SENTIMENT_LABELS_3
    properties = {
        "sentiment": {"type": "string", "enum": labels},
        "rationale": {
            "type": "string",
            "description": "One short sentence explaining the call, for spot-checking only.",
        },
    }
    required = ["sentiment", "rationale"]
    if with_emotion:
        properties["primary_emotion"] = {"type": "string", "enum": EMOTIONS}
        required.append("primary_emotion")

    return {
        "name": "review_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def classify_review(
    title: str,
    text: str,
    n_classes: int = 2,
    with_emotion: bool = False,
    model: str = MODEL,
) -> dict:
    client = get_client()
    system = _system_prompt(n_classes, with_emotion)
    user = f"Title: {title or '(no title)'}\n\nReview: {text or '(no text)'}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": _response_schema(n_classes, with_emotion)},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    spot_checks = [
        ("Amazing, exactly what I needed", "This gift card worked perfectly and the recipient loved it. Fast delivery too!"),
        ("Terrible experience", "Card arrived with zero balance on it. Customer service was no help at all. Waste of money."),
        ("It's a gift card", "Bought it, sent it, used it. Nothing special."),
        ("Fine I guess", "Works as expected but the packaging was a little flimsy. Would probably buy again."),
    ]
    for title, text in spot_checks:
        result = classify_review(title, text, n_classes=3, with_emotion=True)
        print(f"[{result['sentiment']:>8} / {result['primary_emotion']:<11}] {title!r} -> {result['rationale']}")
