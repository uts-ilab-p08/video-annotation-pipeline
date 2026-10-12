# Annotation Validation Report

**Source:** `notebooks/QA/08_validate_annotations.ipynb`, run of 2026-10-11, all 71 videos
**Judge:** `mlx-community/Qwen3-8B-4bit` (local MLX), agreement prompt `v4_unified_scale_judge`
**Pass mark:** judge score of 0.7 or higher (scale 0.0–1.0 in steps of 0.1)
**Row-level data:** `bronze.meva_event_coverage` (Part A, 605 rows) and `bronze.annotations_report`
filtered to `prompt_version = 'v4_unified_scale_judge'` (Part B, 1,952 rows)

## Headline

| Measure | Question | n | Rate |
|---|---|---|---|
| **Coverage** (Part A) | Did the annotations describe each MEVA activity type in the video? | 605 activity/video pairs | **80.5%** |
| Coverage, instance-weighted | Same, weighted by the 2,720 MEVA instances behind the pairs | 2,720 instances | 85.9% |
| **Agreement** (Part B) | Does each generated event match the MEVA activity in its own time window? | 1,785 events | **63.2%** (mean score 0.554) |
| **Unreferenced events** | Generated events with no MEVA activity in their time window | 167 of 1,952 | **8.6%** of events |

## Part A: Coverage

Of 605 MEVA activity/video pairs, 487 are covered. All 118 misses are videos that have annotations, but the
annotations describe something else. No video was missed for lacking annotations.

**Weakest activity types:**

| Activity | Videos | Covered |
|---|---|---|
| person_embraces_person | 7 | 1 |
| person_talks_on_phone | 20 | 7 |
| vehicle_turns_left | 25 | 12 |
| person_talks_to_person | 42 | 22 |
| person_texts_on_phone | 29 | 19 |
| vehicle_reverses | 18 | 12 |

**Strongest activity types:** person_enters_scene_through_structure (48/48), person_exits_scene_through_structure
(43/44), person_transfers_object (19/19), person_stands_up (14/14), and every trunk, loading and unloading activity.

**Videos at 0% coverage:** G301 (2018-03-07 16:50), G506 (2018-03-05 14:05), and G505 (2018-03-07 11:10 and
2018-03-05 13:15). G505 at 11:10 has 30 annotations and still covers none of its 3 activity types.

**Object check:** all 453 annotations credited by the judge name every object type the activity implies.

## Part B: Agreement

Each generated event is paired with the MEVA activities whose frames overlap its window (`FRAME_TOLERANCE = 0`),
and the judge scores it against the best-matching one.

| Status | Events |
|---|---|
| Generated events in scope | 1,955 |
| No MEVA reference in window (not judged) | 167 |
| Judge failed (held back, not written) | 3 |
| **Scored** | **1,785** |
| Matched (score ≥ 0.7) | 1,129 |

**Score distribution:** 1.0: 35 · 0.9: 9 · 0.8: 0 · **0.7: 1,085** · 0.6: 9 · 0.5: 282 · 0.4: 17 · 0.2: 39 · 0.1: 250 · 0.0: 59

**Worst videos:** G508 bus 11:10 (24%), G331 bus 16:50 (28%), G339 school 11:10 (28%), G436 hospital 14:05 (29%).
The lowest scores go to generic movement descriptions, such as "White SUV drives past the camera" or "Person
walks across the grass", made while MEVA records a specific action.

**Prompt versions in `bronze.annotations_report`:**

| Prompt version | Scored | Mean score | Match rate |
|---|---|---|---|
| v2.3_semantic_judge | 739 | 0.557 | 15.3% |
| v2.4_semantic_judge | 1,788 | 0.583 | 19.7% |
| v3_merged_semantic_judge | 1,788 | 0.424 | 15.5% |
| v4_unified_scale_judge | 1,785 | 0.554 | 63.2% |

## Unreferenced events: the 167

These events are not counted for or against agreement. For each one, the gap is measured in frames (30 fps) to the
closest MEVA activity in the same video.

| Distance to nearest MEVA activity | Events | Person movement | Vehicle movement |
|---|---|---|---|
| ≤ 1 s (1–30 frames) | 18 | 12 | 6 |
| 1–3 s | 29 | 13 | 16 |
| 3–5 s | 20 | 14 | 6 |
| 5–10 s | 23 | 11 | 12 |
| 10–30 s | 23 | 15 | 8 |
| > 30 s | 53 | 23 | 30 |
| No ground truth in video (G330 school 16:50) | 1 | 1 | 0 |
| **Total** | **167** | **89** | **78** |

Median gap: 229 frames (7.6 s). Mean: 809 frames (27 s).

- **67 events (40%) are within 5 s** of a MEVA activity, and 47 of them are within 3 s.
- **53 events (32%) are more than 30 s from any MEVA activity.**
- **Most repeated event names:** "White SUV drives past parked vehicles" (5), "Man walks down a staircase" (5),
  "Person walks toward the building entrance" (5), "Person descends stairs" (4), "Person ascends stairs" (3).
- **Most frequent nearest activity:** vehicle_turns_right (33), person_exits_scene_through_structure (30),
  person_enters_scene_through_structure (28), vehicle_starts (15), vehicle_stops (14).
- **Most affected videos:** G505 bus 11:10 (15), G341 hospital 11:10 (11), G340 bus 2018-03-05 13:20 (11),
  G505 bus 11:00 (10).

## Judge sanity check (Part A)

**Coverage score distribution:** 1.0: 42 · 0.9: 4 · 0.8: 1 · **0.7: 440** · 0.6: 38 · 0.5: 8 · 0.0: 72
(7 of 11 score levels used)

The notebook's check flagged 46 annotations credited with covering 3 or more different activity types in the same
video. For example, "A person exits a white SUV and walks away" is credited with 15 activity types in
G424 2018-03-05 13:20.
