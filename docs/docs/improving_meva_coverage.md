# Improving MEVA event coverage

> **Status (2026-09-28):** sections 2.1 to 2.4 and 2.6 shipped in videometa 0.0.17; the first pilot (section 4)
> showed the crop never engaged and the model ignored the checklist. videometa 0.0.18 fixes the crop and the
> second pilot swaps in the 8B model. Section 2.5 (track-proximity windows) is not implemented.
>
> **Design note:** the pipeline is dataset-agnostic. MEVA is only the yardstick, so the action checklist is a
> parameter of the annotators (`action_vocabulary`), not a fixed part of the package.

Diagnosis of the 35% activity coverage reported by `notebooks/QA/08_validate_annotations.ipynb`
(judge `mlx-community/Qwen3-8B-4bit`), and the changes that should raise it, ordered by expected gain.

## 1. Where the events are lost

The coverage table in `bronze.meva_event_coverage` holds 1,318 evaluable (video, activity type) pairs:
460 covered, 858 not (34.9%). A further 10,018 pairs are `no_video`: MEVA has ground truth for 761
videos and only 72 have been through the pipeline. Those are excluded from the rate but should be stated
whenever the number is reported.

The 858 misses were traced stage by stage against the local data in `data/interim` and `data/processed`.

| Stage | Measurement (72 videos) | Verdict |
|---|---|---|
| Motion gate (nb 04) | 91% of (video, activity type) pairs touch at least one window; 79% of MEVA instances touch a window, 63% are at least half inside one. Windows cover 132 s of each 300 s video on average. | Secondary loss. Worst for short person-vehicle actions: `person_closes_vehicle_door` 57%, `person_opens_vehicle_door` 64%, `person_exits_vehicle` 64%, `person_unloads_vehicle` 50%. |
| VLM description (nb 05) | 70% of windows return exactly one event (1.32 events per window). Of 1,218 events, 0% mention reversing, 1% a trunk or boot, 2% talking, 3% turning, 3% entering or exiting a vehicle. The most common event names are "Person walks across the room / along a sidewalk / across the parking lot". | **Primary loss.** The model summarises a 10 s window as one "person walks" event and never names the fine-grained action MEVA records. |
| Output truncation (nb 05) | 98 of 922 windows (10.6%) failed with `Unterminated string ... char ~2450`. Notebook 05 sets `max_tokens=600` (the package default is 1200) while the prompt asks for 4 to 7 sentences per event plus 1 to 2 sentences per involved object. | Every multi-event window that overruns 600 tokens is discarded whole. |
| Visual input | Source is 1920x1080. Notebook 04 has `WindowSpatialFeatureJoiner` shrink the whole frame to 1280x720 (1.5x; a few older windows on disk are at the package default 640x360, 3x). Median tracked person height 206 px becomes 137 px and the 10th percentile 71 px becomes 47 px; cars go from 118 px to 79 px. At 2 fps a 2 s door action is 4 frames. | Door, phone, talking and trunk actions on the smaller, more distant subjects are not resolvable at that size, and the tracked subjects occupy a small part of the frame the model is looking at. |
| Judge (nb 08) | Scores used: 0.0 (858), 0.75 (70), 1.0 (390). Reasons are specific: "all annotations describe people walking", "only describe vehicles moving left to right". | Strict by design and consistent with the vocabulary counts above. Not the bottleneck. |

The coverage by activity type in the database says the same thing: the activities the pipeline does cover
are the ones "person walks" already describes (`person_enters_scene_through_structure` 82%,
`person_exits_scene_through_structure` 82%), and the ones it misses are actions
(`vehicle_turns_right` 0/66, `person_talks_to_person` 9/88, `person_opens_facility_door` 16/82,
`person_puts_down_object` 7/51, `vehicle_reverses` 3/41, `person_closes_vehicle_door` 5/41).

## 2. Changes, in the order to make them

### 2.1 Stop losing windows to truncation (notebook 05 + package)

In `05_annotate_prepared_windows.ipynb`, raise the output budget:

```python
annotator = LocalQwenEventAnnotator(
    model_id=MODEL_ID,
    video_fps=2.0,
    max_tokens=1500,          # was 600; the prompt asks for far more than 600 tokens of JSON
    prompt_token_budget=6000,
    feature_frames=8,
)
```

In `videometa/window_annotation.py`, salvage the complete events when the model still runs out of
tokens instead of raising and losing the window. Replace the `json.loads(_strip_json_fence(output_text))`
call in `LocalQwenEventAnnotator._generate_events` (and the same line in `LVLMEventAnnotator.annotate`)
with `_parse_event_list`:

```python
def _parse_event_list(content: str) -> list[Any]:
    """Parse the model's JSON, keeping every complete event when the output was cut off."""
    cleaned = _strip_json_fence(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = _salvage_events(cleaned)
        if not parsed:
            raise
        logger.warning("Model output was truncated; salvaged %d complete events", len(parsed))
    if isinstance(parsed, dict):
        parsed = parsed.get("events")
    if not isinstance(parsed, list):
        raise ValueError("Model response must be a JSON event list.")
    return parsed


def _salvage_events(text: str) -> list[dict[str, Any]]:
    """Decode the leading complete objects of a JSON array whose tail is missing."""
    start = text.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    position = start + 1
    events: list[dict[str, Any]] = []
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text) or text[position] != "{":
            break
        try:
            event, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        events.append(event)
    return events
```

Expected effect: the 10.6% of windows currently lost come back, and the 1,500 budget lets the model list
more than one event per window.

### 2.2 Make the model enumerate actions, not narrate the window (package prompt)

This is the change that targets the 858 misses directly. `DESCRIPTION_GUIDANCE` currently optimises for
one richly written story per event ("four to seven complete sentences"), so the model writes one event per
window about the most visible movement. The fix is to require one event per (subject, action) and to hand
the model a controlled action vocabulary, written in plain English and derived from the MEVA activity
list, so that the words the judge is looking for are the words the model is asked to use.

Add to `window_annotation.py`:

```python
#: Plain-English actions the annotator must look for and name when present. Derived from the MEVA
#: activity taxonomy, so a downstream comparison against MEVA ground truth uses the same vocabulary.
ACTION_VOCABULARY = (
    # people and vehicles
    "opens a vehicle door", "closes a vehicle door", "gets into a vehicle", "gets out of a vehicle",
    "opens a trunk or tailgate", "closes a trunk or tailgate", "loads something into a vehicle",
    "unloads something from a vehicle",
    # people
    "talks to another person", "talks on a phone", "texts or looks at a phone", "reads a document",
    "picks up an object", "puts down an object", "carries a heavy object", "hands an object to another person",
    "hugs another person", "opens a building door", "closes a building door", "enters through a doorway",
    "exits through a doorway", "sits down", "stands up", "rides a bicycle", "walks through the scene",
    # vehicles
    "vehicle turns left", "vehicle turns right", "vehicle makes a U-turn", "vehicle stops",
    "vehicle starts moving", "vehicle reverses", "vehicle drops off a person", "vehicle picks up a person",
    "vehicle drives through the scene",
)

ACTION_GUIDANCE = (
    "Before writing, go through this checklist for EVERY person and EVERY vehicle in the window and "
    "decide which of these actions you can actually see: "
    + "; ".join(ACTION_VOCABULARY)
    + ".\n"
    "Return one event per subject per action. A person who gets out of a car, closes its door and then "
    "talks to someone is three events, not one. Ordinary walking or driving is also an event, but never "
    "let it stand in for a more specific action that is visible in the same window.\n"
    "Each event carries an actions list holding the checklist phrases it shows, copied exactly, and the "
    "description must state each of those actions in words. Do not list an action you cannot see."
)
```

Then in `DESCRIPTION_GUIDANCE`, change the description rule from "four to seven complete sentences, richly
detailed" to "two to four sentences" and cut `PHYSICAL_DETAILS_GUIDANCE` to one short phrase per object.
The prose is not what the coverage check reads; the action is. In `_build_window_prompt` (and
`LVLMEventAnnotator._build_prompt`) insert `ACTION_GUIDANCE` after `DESCRIPTION_GUIDANCE` and add the field
to the schema:

```python
'Return JSON only as a list of events: [{"event_name": str, "description": str, '
'"actions": [str], "involved_objects": [{"id": str, "label": str, "physical_details": str}]}].'
```

In `_clean_events`, normalise `actions` to the vocabulary (lowercase, strip, keep phrases that are in
`ACTION_VOCABULARY`, keep others under a separate `other_actions` key so nothing is silently lost).

Downstream, the judge only sees `event_name` and `description` from `bronze.events`. Either carry
`actions` through notebook 06 and 07 as a new column and add it to `numbered_annotations()` in notebook 08,
or, with no schema change, have notebook 06 append `Actions: opens a vehicle door; gets into a vehicle.`
to the end of the exported description. The column is cleaner and also gives the precision report a
structured field to match on.

### 2.3 Crop the window to the activity instead of shrinking the whole frame (package)

`_write_annotated_window_video` resizes the whole 1920x1080 frame (to 1280x720 in notebook 04, 640x360 by
default), which is what shrinks a 206 px person to 137 px or 69 px while most of the picture is empty car park. The tracked boxes for the window are already in `frame_annotations`, so the joiner can crop to
the region where the tracked people and vehicles are, then resize that region to 640x360. Token cost to
the VLM is unchanged because the output video is the same size.

Sketch, in `window_annotation.py`:

```python
_CROP_LABELS = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


def _activity_crop(
    frame_annotations: Sequence[Any], frame_size: tuple[int, int], aspect: float, pad: float = 0.2
) -> tuple[int, int, int, int] | None:
    """Smallest 16:9 region holding every tracked person or vehicle in the window, or None for the full frame."""
    width, height = frame_size
    boxes = [
        d.boundary for f in frame_annotations for d in f.detections if d.label in _CROP_LABELS
    ]
    if not boxes:
        return None
    left, top = min(b.left for b in boxes), min(b.top for b in boxes)
    right, bottom = max(b.right for b in boxes), max(b.bottom for b in boxes)
    crop_w, crop_h = (right - left) * (1 + pad), (bottom - top) * (1 + pad)
    if crop_w * crop_h > 0.6 * width * height:
        return None  # the action fills the frame already; cropping buys nothing
    # grow the shorter side to the target aspect ratio, then clamp to the frame
    if crop_w / crop_h < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0 = int(max(0, min(cx - crop_w / 2, width - crop_w)))
    y0 = int(max(0, min(cy - crop_h / 2, height - crop_h)))
    return x0, y0, int(min(width, x0 + crop_w)), int(min(height, y0 + crop_h))
```

In `_write_annotated_window_video`, compute the crop once per window, slice each frame with it before
`cv2.resize`, and pass the offset into `_draw_detections` (it currently takes only `scale_x`/`scale_y`;
add `offset_x`/`offset_y` and subtract them before scaling). Expose it as
`WindowSpatialFeatureJoiner(crop_to_activity=True)` so notebook 04 can switch it on. If one window holds
two far-apart groups, the 60% guard falls back to the full frame; a second iteration could split such a
window into one crop per group.

Also worth trying in the same pass: `annotated_video_fps=3.0` with `annotated_video_max_frames=36`, so a
2 s door action is 6 frames instead of 4.

### 2.4 Loosen the motion gate (notebook 04 config only)

Re-cutting windows from one motion pass on the three videos with the worst gate recall:

| Config | G328 13:20 (104 inst.) | G341 11:00 (112 inst.) | G339 11:00 (126 inst.) | Window seconds |
|---|---|---|---|---|
| current: `k=1.5, pad 2, gap 3` | 52% | 59% | 62% | 125 to 134 s |
| `k=1.0, pad 3, gap 5` | 66% | 71% | 72% | 167 to 178 s |
| `k=0.5, pad 3, gap 5` | 76% | 74% | 78% | 197 to 199 s |

Activity types touched moved less (G328: 14 to 16 of 18), which is why this is fourth: the metric is
per activity type and 91% of pairs already touch a window. It still matters for the instance-weighted
number and for short door and trunk actions. In `04_prepare_video_windows.ipynb`:

```python
motion_config = MotionGateConfig(
    motion_threshold="auto",
    motion_std_k=0.5,          # was 1.5
    pad_seconds=3.0,           # was the 2.0 default
    merge_gap_seconds=5.0,     # was the 3.0 default
    warmup_seconds=2,
    mog_history=100,
    min_window_seconds=4.0,
    max_window_seconds=12.0,
    max_total_seconds=200.0,   # cap the extra VLM cost per video
    spatial_diversity=True,    # spend the budget across the frame, not on the road twice
)
```

This is about 1.5x the VLM time per video. Reproduce the table with
`RelevantWindowFinder(config).sample_motion()` once and `build_windows()` per config; the script used for
the table is small enough to become a cell in notebook 04.

### 2.5 Track-proximity windows (package, larger change)

`ObjectBoundaryExtractor.extract` already runs YOLO over every frame of the video and only keeps the
frames that fall in a window. The lowest-recall MEVA classes (door open/close, enter/exit vehicle, trunk,
load/unload) are exactly "a person next to a stationary vehicle", which produces little motion. A cheap
second gate is: for every person track whose box overlaps or comes within one box-width of a vehicle
track for at least 1 s, add a window around that span (padded 3 s, merged with neighbours) and mark it
relevant. It needs `extract` to keep per-frame detections for the whole video (9,000 frames of a few boxes
each is small) and to accept extra windows after tracking, so it is a refactor of the stage order:
track once, then build motion windows plus proximity windows, then assign frames. Do this after 2.1 to 2.4
have been measured; it is the piece with the most engineering in it.

### 2.6 Judge and reporting

Keep the rules; they are doing their job. Two small changes: show the judge the new `actions` field and
`involved_objects[].physical_details` (the person's clothing often carries "holding a phone"), and run a
100-pair agreement check between the local Qwen3-8B judge and Groq `openai/gpt-oss-120b` before reporting
a new number, since section 10.2 of notebook 08 only checks for rubber-stamping, not for over-strictness.
When quoting the coverage rate, also quote that 689 of the 761 MEVA videos have not been processed.

## 3. Pilot 2: what to run

videometa 0.0.18 is in the local source tree on `feature/annotations`. Releases are made by the GitHub
workflow, not by hand: merging a pull request into `main` bumps the patch version in `pyproject.toml`
(main is at 0.0.17, so the merge produces 0.0.18), builds, publishes to TestPyPI and tags `v0.0.18`. Do not
edit the version line on the branch, or the merge conflicts with the bot's bump commit.

```bash
cd "/Users/juanvargas/UTS/3rd Semester/Capstone/videometa"
uv run pytest -q tests/test_videometa.py        # 53 pass; the 3 failures pre-date these changes
git add -A && git commit -m "Actor crop, configurable action vocabulary, duplicate events dropped"
git push origin feature/annotations
# open a PR feature/annotations -> main and merge it; wait for the "Publish to TestPyPI" workflow
cd "/Users/juanvargas/UTS/3rd Semester/Capstone/annotation_pipeline"
uv lock --upgrade-package videometa && uv sync
uv run python -c "import videometa; print(videometa.__version__)"   # expect 0.0.18
```

The notebooks are set to one video, `2018-03-05.13-20-01.13-25-01.school.G328` (104 MEVA instances, 18 activity
types, 19 windows), so a pilot is about 20 VLM calls. Run 04, 05, 06, 07, 08 in order without changing anything.

| Notebook | Setting |
|---|---|
| 04 | `VIDEO_FILTER = "<that video>"`, `OVERWRITE = True`, crop on (actor rule) |
| 05 | `VIDEO_FILTER = "<that video>"`, `OVERWRITE = True`, `MODEL_ID` = Qwen3-VL-**8B**-Instruct-4bit |
| 06 | `OVERWRITE = False` |
| 07 | `VIDEO_FILTER = "<that video>"`: deletes and reloads only that video's rows; `None` empties the bronze tables and reloads every export |
| 08 | `VIDEO_FILTER = "<that video>"`: judges only that video and replaces only its rows in `bronze.meva_event_coverage`; `None` re-judges everything and rebuilds the table |

`VIDEO_FILTER` is a substring match on the video name, so a camera id selects a
camera and `None` selects everything. Two things to read afterwards: `crop_box` in the video's
`prepared_windows.json` should now be set on most windows, and the 08 by-activity table for that video against
the pilot 1 numbers in section 4.

To separate the two changes, run 05 once more with the 4B `MODEL_ID` line swapped in: same cropped windows,
different model.

## 4. Pilot 1 result (videometa 0.0.17, 4B model, camera G328)

Coverage did not improve: 9 of 66 pairs (13.6%) against roughly 31% expected for that camera's activity mix.
What the run showed:

- Truncation fixed: 1 window lost of 32, against 10.6% before.
- 4.22 events per window against 1.32 before, but 48 of the 135 events were exact duplicates.
- The crop engaged on 0 of 71 windows: the union of all tracked people and vehicles in a car park is the whole
  frame. Recomputed offline, a moving-tracks rule crops 40 of 71.
- The model wrote its own phrases in `actions` in 128 of 135 events and never mentioned a door, trunk, phone,
  conversation, turn or reverse in any description.

Section 3 is the response: the actor crop and a larger model, with everything else held constant.

## 5. Inspecting verdicts: the coverage explorer

`reports/coverage_explorer.html` shows, per video, every MEVA activity type with its verdict next to the
pipeline's annotations, plays the video fragment MEVA points at, and draws both sets of boxes over the video:
MEVA actor boxes in amber (from the raw Kitware geom files, which are not in the database) and the pipeline's
YOLO boxes in teal. Filter by covered / not covered, click an instance to loop its fragment, and the compare
strip shows the activity beside the annotation the judge credited, with the judge's reason.

```bash
make coverage-explorer      # exports reports/coverage_explorer_data/ from bronze, then serves the project root
# open http://localhost:8000/reports/coverage_explorer.html
```

The export takes the newest judge model's verdicts, one per pair, so rerun it after notebook 08. Video plays
from the public R2 URL and falls back to `data/processed/source_mp4/<video_id>.mp4` when that fails.

