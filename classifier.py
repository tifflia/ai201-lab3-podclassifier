import json
import os
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    The prompt has three parts:
      1. Task instruction describing the four valid labels.
      2. The labeled examples (omitted entirely when none are provided, which
         turns this into a zero-shot prompt driven by the definitions alone).
      3. The new episode to classify, in the same format as the examples, plus
         the requested output format.

    The LLM is asked to respond as a JSON object:
        {"label": <one of the four labels>, "reasoning": <brief explanation>}
    classify_episode() pairs this with response_format={"type": "json_object"}
    and parses it with json.loads(). (That enforced mode requires the word
    "JSON" to appear in the prompt, so the output instruction states it.)
    """
    instruction = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Return only the label and your reasoning. Do not explain the taxonomy."
    )

    parts = [instruction]

    if labeled_examples:
        example_blocks = []
        for ex in labeled_examples:
            example_blocks.append(
                f"Title: {ex['title']}\n"
                f"Description: {ex['description']}\n"
                f"Label: {ex['label']}"
            )
        parts.append("Examples:\n\n" + "\n\n---\n\n".join(example_blocks))
    else:
        parts.append(
            "No labeled examples are provided; classify using the definitions above."
        )

    parts.append(
        "Classify the episode below. If the description is brief, infer the "
        "most likely format from available cues and note your uncertainty in "
        "the reasoning.\n\n"
        f"Title: ?\n"
        f"Description: {description}\n\n"
        "Respond with a single JSON object and nothing else, in exactly this shape:\n"
        '{"label": "<one of: interview, solo, panel, narrative>", '
        '"reasoning": "<brief explanation>"}'
    )

    return "\n\n".join(parts)


def _parse_response(text: str) -> tuple[str, str]:
    """
    Extract (label, reasoning) from the LLM's JSON response.

    The model is asked for {"label": ..., "reasoning": ...}, but may wrap it in
    ```json fences or add a preamble, so we pull out the first {...} block before
    parsing. Raises JSONDecodeError if no valid JSON object is present — the
    caller treats that as a failure and returns "unknown".
    """
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    data = json.loads(text)
    return str(data.get("label", "")).lower(), str(data.get("reasoning", ""))


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Returns a dict with "label" (one of VALID_LABELS, or "unknown" on an invalid
    or failed classification) and "reasoning". Never raises — a single bad
    response yields one "unknown" row rather than crashing the evaluation loop.
    """
    # Guard: an empty description can't be classified — don't spend an API call.
    if not description or not description.strip():
        return {"label": "unknown", "reasoning": "empty description"}

    try:
        prompt = build_few_shot_prompt(labeled_examples, description)
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        print(text) # DEBUG
        label, reasoning = _parse_response(text)
    except Exception as e:
        return {"label": "unknown", "reasoning": f"error: {e}"}

    label = label.strip().lower()
    if label not in VALID_LABELS:
        label = "unknown"

    return {"label": label, "reasoning": reasoning}
