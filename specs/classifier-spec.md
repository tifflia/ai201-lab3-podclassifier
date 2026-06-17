# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
JSON with response_format={"type": "json_object"}. It maps directly onto the required return dict and the enforced mode removes the fence/preamble failure mode that makes JSON annoying otherwise.
```

---

**Edge cases to handle in the prompt:**

```
1. labeled_examples is empty
- Build conditionally. If there are no examples, omit the "Examples:" header and the example block entirely rather than emitting an empty section or a dangling ---. A header with nothing under it is confusing and can make the model think examples were lost.
- Keep the task instruction and the new-episode block exactly as-is. The definitions do the work.
- Optionally add one line like "No labeled examples are provided; classify using the definitions above." so the model doesn't hunt for examples that aren't there.

2. Very short description
- Still require a single label. Instruct the model to make its best guess from whatever cues exist and explain the uncertainty in the reasoning field. Don't invite "unknown" here: per the spec, "unknown" is reserved for invalid/error cases (classifier-spec.md:128), not low confidence. Mixing the two pollutes your eval.
- A guiding line helps: "If the description is brief, infer the most likely format from available cues and note your uncertainty in the reasoning."

3. Empty or missing description
- Guard before the LLM call in classify_episode(): if description.strip() is empty, short-circuit to {"label": "unknown", "reasoning": "empty description"} without calling the model. This is a genuine "can't classify" case, so "unknown" is correct here — and it keeps a bad input from costing a request or producing a hallucinated label.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
The model returns {"label": "interview", "reasoning": "..."}, possibly wrapped in ```json fences if you didn't enforce response_format.

import json, re

def _parse_response(text):
    text = text.strip()
    # strip ```json ... ``` fences if present
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    data = json.loads(text)            # may raise JSONDecodeError
    return str(data.get("label", "")).lower(), str(data.get("reasoning", ""))
```

---

**Step 4 — Validate the label:**

```
After parsing, check whether the extracted label is in VALID_LABELS.
Normalize first — strip surrounding whitespace and lowercase it — so that
"Interview", " interview", or "interview." compare correctly against the
lowercase labels in VALID_LABELS.

If the normalized label is in VALID_LABELS, keep it.
If it is not (the model returned an off-taxonomy word, an empty string from
a failed parse, or extra text), set label to "unknown".

Do NOT pass an unvalidated label through to the return dict — the contract
guarantees "label" is either one of VALID_LABELS or "unknown", and the
evaluation loop relies on that.

label = label.strip().lower()
if label not in VALID_LABELS:
    label = "unknown"
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the LLM call and parsing in a try/except so that no single failure
can crash the 20-call evaluation loop. The whole flow — building the
prompt, calling the API, and parsing the response — happens inside the try.

What can go wrong:
  - Network / API errors: timeout, connection drop, rate limit (429),
    auth failure, or the server returning a 5xx.
  - Empty or malformed response: the model returns nothing, or text that
    the parser can't extract a label/reasoning from (e.g. JSONDecodeError
    if using JSON, or a response with no "Label:" line).
  - Unexpected response shape: response.choices is empty, so indexing
    [0].message.content raises IndexError/AttributeError.

On ANY failure, return a valid dict rather than raising:
  {"label": "unknown", "reasoning": "<short description of what failed>"}
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: The Aral Sea: A Disaster in Four Acts
Raw response text: {
  "label": "narrative",
   "reasoning": "The episode tells a story in multiple parts, including historical context, economic collapse, and unexpected recovery, which suggests an assembled narrative with a clear arc, drawing on external sources."
}
```

**How did you parse the label out of the response?**

```
The response is JSON, so it's parsed as JSON rather than with raw string splitting. In _parse_response():
  1. text.strip() to drop surrounding whitespace.
  2. re.search(r"\{.*\}", text, re.DOTALL) to pull out the first {...} block,
     in case the model wraps the object in ```json fences or adds a preamble.
  3. json.loads() on that block to get a dict.
  4. data.get("label", "") to read the label field, wrapped in str() and
     .lower() so it's a lowercased string.
Then in classify_episode() the label is .strip().lower()'d again and checked against VALID_LABELS; anything not in the set (or any parse/API error) becomes "unknown".
```

**Did any episodes return `"unknown"`? If so, why?**

```
no
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
