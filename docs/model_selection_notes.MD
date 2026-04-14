# Model Comparison (Few-Shot)

## What “Exact Match” Means
Exact match = model output must be **identical** to the correct answer.  
Even one character off = counted as wrong.

---

## Results (Test Split, Few-Shot)

### Meter Task
- Best model: FLAN-T5-Large
- Score: 34.8%
- Gets the full answer exactly right about 1 in 3 times

### Rhyme Task
- All models: 0% exact match
- No model ever produced a perfect full answer
- Small signal: FLAN-T5-Small gets partial matches rarely

### Natural Text Task
- All models: 0%
- Matching full lines exactly is too strict

### Combined Task
- All models: 0% exact match
- GPT-2 does best on partial correctness (~29.6%)

---

## Overall Ranking (Mean Exact Match)

| Model | Mean Exact % |
|-------|-------------|
| FLAN-T5-Large | 8.7% |
| FLAN-T5-Small | 1.21% |
| Others | ~0% |

---

## Key Takeaway

Exact match is very strict.  
It only works for simple tasks (like meter).  
For longer or structured outputs, it hides differences between models.

---

## Data Source

Results come from:
`docs/model_comparison.csv`
