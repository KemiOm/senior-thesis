# Data Overview — 18th-Century Poetry Corpus

Running document for the thesis preprocessing pipeline and database structure.

---

## 1. Layers

| Layer | Holds |
|-------|-------|
| **Poem** | `id`, `author`, `title`, source (ECPA), optional metadata |
| **Stanza** | `index`, reference to poem; used for rhyme schemes |
| **Line** | `index`, `raw_text`, `normalized_text`, annotations |

---

## 2. Per-Line Annotation Fields

| Field | Purpose | Populated By |
|-------|---------|--------------|
| `raw_text` | Original text from XML | Extraction (done) |
| `normalized_text` | Quotes, dashes, whitespace cleaned | Normalization |
| `rhyme_word` | Word that rhymes | Rhyme detection |
| `rhyme_group` | Rhyme pair (A, B, C…) | Per-stanza rhyme scheme |
| `meter` | e.g. iambic pentameter | Poesy / rule-based |
| `stress_pattern` | Syllable stresses | CMU/Poesy |
| `end_stopped` | Punctuation at line end | Punctuation check |
| `enjambment` | Continues to next line | Punctuation + boundaries |
| `caesura` | Midline break position | Punctuation / semantic |
| `phonology` | Pronunciations per word | CMU dict, Poesy, espeak |

---

## 3. Phonology (Per-Word)

```json
"phonology": [
  { "word": "Friend", "arpabet": ["F R EH1 N D"], "source": "cmudict" },
  { "word": "lov'd", "arpabet": ["L AH1 V D"], "source": "poesy" }
]
```

- **source**: `cmudict`, `poesy`, `espeak`, `manual`
- CMU: ARPAbet, stress on vowel (0/1/2)
- Fallback for unknown words: espeak

---

## 4. Storage

| Stage | Path | Contents |
|-------|------|----------|
| Extracted (raw) | `output/poems/<id>.json` | `id`, `author`, `title`, `stanzas` (lists of line strings) |
| Normalized | `output/poems_normalized/<id>.json` | Same + each line as `{raw, normalized}` |
| Annotated | `output/poems_annotated/<id>.json` | + `phonology`, `rhyme_word`, `rhyme_group`, `meter`, `stress`, `end_stopped`, `caesura`, `enjambment`, `stanza_index`, `line_index`; stanza-level `stanza_type`, `rhyme_scheme`, `rhyme_pairs` |
| SQLite | `output/corpus.db` | `poems`, `stanzas`, `lines` tables for querying |

---

## 5. Pipeline Order

1. **Extraction** ✓
2. **Normalization** (quotes, dashes, whitespace) ✓
3. **Phonology** (CMU + Poesy) ✓
4. **Stress & meter** ✓
5. **Rhyme** (rhyme word, rhyme group) ✓
6. **Punctuation cues** (end_stopped, caesura) ✓
7. **Enjambment** ✓

---

## 6. File Layout

| Script | Purpose |
|--------|---------|
| `extract_sample.py` | Single-poem extraction |
| `extract_batch.py` | Batch extraction → `output/poems/` |
| `normalize_sample.py` | Single-poem normalization |
| `normalize_batch.py` | Batch normalization → `output/poems_normalized/` |
| `phonology_sample.py` | Single-poem annotation (Poesy/Prosodic, CMU, punctuation); runs quatrain + couplet samples |
| `phonology_batch.py` | Batch annotation → `output/poems_annotated/` |
| `export_sqlite.py` | Export annotated JSON → `output/corpus.db` |

**Install for step 3 (phonology):**
```bash
brew install espeak
pip install pronouncing
pip install git+https://github.com/quadrismegistus/poesy
```
