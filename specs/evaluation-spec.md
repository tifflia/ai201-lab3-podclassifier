# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = (number of predictions that exactly match the ground truth)
           / (total number of predictions)

A single prediction counts as "correct" when predictions[i] == ground_truth[i] (exact string match, same position).
```

---

**Step-by-step logic:**

```
1. If predictions is empty, return 0.0 (see edge case below).
2. Initialize a counter `correct = 0`.
3. Walk the two lists in parallel (e.g. zip(predictions, ground_truth)). For each pair, if predicted == truth, increment `correct`.
4. Return correct / len(predictions) as a float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. With no episodes there are no correct predictions and dividing by len == 0 would raise ZeroDivisionError, so we guard against it. 0.0 is a safe, sensible default ("nothing was classified correctly out of nothing").
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Compare position by position:
  index 0: "interview" == "interview"  ✓ correct
  index 1: "solo"      == "solo"       ✓ correct
  index 2: "panel"     != "solo"       ✗ wrong
  index 3: "interview" != "narrative"  ✗ wrong

correct = 2, total = 4
compute_accuracy() returns 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
We measure per-class accuracy against the ground-truth label. An episode counts toward the "interview" class only when its ground_truth is "interview". It is "correct" for that class when ground_truth == "interview" AND the prediction also equals "interview" (the model got that true-interview right).

Predictions that wrongly say "interview" for a non-interview episode do NOT count here — they belong to whatever class their ground_truth actually is.
```

---

**What does "total" mean for a given class?**

```
"total" is class-specific, not the total number of predictions. For a given label it is the number of episodes whose ground_truth equals that label — i.e. how many true examples of that class appear in the test set. The four per-class totals therefore sum to len(ground_truth).
```

---

**Step-by-step logic:**

```
1. Initialize a stats dict with an entry for every label in VALID_LABELS:
   {label: {"correct": 0, "total": 0, "accuracy": 0.0}}.
2. Loop over the (predicted, truth) pairs in parallel (zip).
3. For each pair, bucket it by its ground-truth label:
     - increment stats[truth]["total"]
     - if predicted == truth, also increment stats[truth]["correct"]
   (Guard: if truth somehow isn't in VALID_LABELS, skip it.)
4. After the loop, compute each class's accuracy: for every label, if
   total > 0 set accuracy = correct / total, otherwise leave it 0.0.
5. Return the stats dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0 (as the docstring in evaluate.py specifies: "accuracy: correct / total (0.0 if total is 0)"). With no ground-truth examples of that class we can't compute correct / total without dividing by zero, so we fall back to 0.0. correct and total both stay 0, which signals "no data for this class" rather than "perfect" — important not to report it as 1.0.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Bucket each pair by its ground-truth label:
  index 0: truth=interview, pred=interview  → interview: total+1, correct+1
  index 1: truth=solo,      pred=interview  → solo:      total+1
  index 2: truth=solo,      pred=solo       → solo:      total+1, correct+1
  index 3: truth=panel,     pred=panel      → panel:     total+1, correct+1
  index 4: truth=narrative, pred=panel      → narrative: total+1

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.0
solo           1       2      0.5
panel          1       1      1.0
narrative      0       1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
