# MEVA Validation Pipeline

This folder contains the scripts used to validate generated CCTV events against the MEVA ground-truth annotations.

## Structure

```text
Validation/
├── fetch_meva.py
├── pasrse_meva.py
├── meva_ontology.py
├── map_to_meva.py
├── match_events.py
├── match_events_semantic.py
├── match_events_final.py
├── diagnose_matching.py
└── diagnose_final_matches.py

data/
├── raw/
│   └── annotations/
│       ├── kitware/
│       │   └── meva_sparse/
│       └── meva_original_annotations/
│
└── interim/
    ├── meva_reference/
    │   └── meva_events.csv
    │
    └── validation/
        ├── event_matches.csv
        ├── semantic_event_matches.csv
        ├── generated_meva_mapping.csv
        └── final_event_matches.csv
```

## Pipeline

```text
videos.csv
    ↓
fetch_meva.py
    ↓
MEVA YAML annotations
    ↓
pasrse_meva.py
    ↓
meva_events.csv
    ↓
map_to_meva.py + meva_ontology.py
    ↓
generated_meva_mapping.csv
    ↓
match_events_final.py
    ↓
final_event_matches.csv
```

## Scripts

| Script                        | Purpose                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `fetch_meva.py`             | Fetches required MEVA YAML annotations using sparse checkout.  |
| `pasrse_meva.py`            | Parses MEVA activity YAML files into`meva_events.csv`.       |
| `meva_ontology.py`          | Defines the MEVA activity classes used for mapping.            |
| `map_to_meva.py`            | Maps generated free-text events to a MEVA activity or`NONE`. |
| `match_events.py`           | Initial exact-label matching experiment.                       |
| `match_events_semantic.py`  | Experimental semantic-similarity matcher.                      |
| `match_events_final.py`     | Final activity + temporal IoU + Hungarian matcher.             |
| `diagnose_matching.py`      | Diagnostic tool for earlier matching experiments.              |
| `diagnose_final_matches.py` | Shows same-class MEVA candidates and IoU for mapped events.    |

## Reporting Files

### `generated_meva_mapping.csv`

Use this for **ontology-mapping analysis**.

Important columns:

- `event_name` — generated event
- `mapped_meva_activity` — corresponding MEVA class
- `mapping_similarity` — mapping confidence
- `mapping_reason` — why it was mapped/rejected
- `NONE` — generated event could not be reliably aligned with a MEVA activity

Current result:

```text
Generated events: 123
MEVA mapped:       4
NONE:            119
```

### `final_event_matches.csv`

Use this for **final MEVA validation reporting**.

Read `match_status` as:

| Status | Meaning                                            |
| ------ | -------------------------------------------------- |
| `TP` | Correct activity and temporal IoU passes threshold |
| `FP` | Generated event has no valid MEVA match            |
| `FN` | MEVA event has no valid generated match            |

Use this CSV for **TP, FP, FN and temporal IoU**. Precision, Recall and F1 are printed by `match_events_final.py`.

## IoU Threshold

Modify the threshold in:

```text
Validation/match_events_final.py
```

Change:

```python
MIN_IOU = 0.50
```

Then rerun:

```bash
python Validation/match_events_final.py
```

This regenerates `final_event_matches.csv`.

## Main Reporting Flow

```text
generated_meva_mapping.csv
→ How well generated events align with MEVA taxonomy

final_event_matches.csv
→ How well generated events match actual MEVA ground truth
```

Use **both CSVs** in reporting: the first explains ontology alignment, while the second provides the final event-level validation results.
