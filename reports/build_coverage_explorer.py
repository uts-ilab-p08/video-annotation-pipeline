"""Export the data behind reports/coverage_explorer.html.

One JSON file per video under reports/coverage_explorer_data/, plus index.json.
Each file joins three sources for one video:

* MEVA ground truth: activities and per-frame actor boxes from the raw Kitware
  YAML in data/raw/annotations/kitware/ (the boxes are not in the database).
* The pipeline's annotations: bronze.events, their objects and per-frame YOLO
  boxes from bronze.geometries.
* The coverage verdicts: bronze.meva_event_coverage, newest verdict per pair (Part A, recall).
* The agreement verdicts: bronze.annotations_report, newest prompt version, one per
  pipeline event (Part B, precision).

Run `make coverage-explorer-data` (or `.venv/bin/python reports/build_coverage_explorer.py`)
for every video, or pass `--video <substring of the name or the video id>` to
rebuild one video's file and update index.json in place. `make coverage-explorer`
serves the page through reports/serve_coverage_explorer.py, which calls
`build()` on demand so the page always reflects the database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KITWARE_DIR = PROJECT_ROOT / "data" / "raw" / "annotations" / "kitware"
LOCAL_MP4_DIR = PROJECT_ROOT / "data" / "processed" / "source_mp4"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "coverage_explorer_data"

#: Keep every Nth MEVA box. MEVA boxes exist for every frame at 30 fps; the
#: overlay only needs a box near the current playback time.
MEVA_BOX_STRIDE = 2


_ENGINE = None


def make_engine():
    """One pooled engine per process: the explorer server calls build() many times, and a new
    connection to the Supabase pooler costs about three seconds from here."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("SUPABASE_DB_URL")
    if url:
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    else:
        url = URL.create(
            drivername="postgresql+psycopg2",
            username=os.environ.get("SUPABASE_DB_USER", "postgres.uqofqjychtjsvfioebod"),
            password=os.environ["SUPABASE_DB_PASSWORD"],
            host=os.environ.get("SUPABASE_DB_HOST", "aws-0-ap-south-1.pooler.supabase.com"),
            port=int(os.environ.get("SUPABASE_DB_PORT", "5432")),
            database=os.environ.get("SUPABASE_DB_NAME", "postgres"),
        )
    _ENGINE = create_engine(url, pool_pre_ping=True)
    return _ENGINE


def read_kitware(video_name: str):
    """Activities (by MEVA id2), actor types and subsampled actor boxes for one video."""
    activities: dict[int, dict] = {}
    path = KITWARE_DIR / f"{video_name}.activities.yml"
    if path.exists():
        for item in yaml.safe_load(path.read_text()) or []:
            act = item.get("act") if isinstance(item, dict) else None
            if not act:
                continue
            span = act["timespan"][0]["tsr0"]
            activities[int(act["id2"])] = {
                "type": next(iter(act["act2"])),
                "start_frame": int(span[0]),
                "end_frame": int(span[1]),
                "actors": [int(actor["id1"]) for actor in act.get("actors", [])],
            }

    types: dict[int, str] = {}
    path = KITWARE_DIR / f"{video_name}.types.yml"
    if path.exists():
        for item in yaml.safe_load(path.read_text()) or []:
            rec = item.get("types") if isinstance(item, dict) else None
            if rec:
                types[int(rec["id1"])] = next(iter(rec["cset3"]))

    wanted = {actor for act in activities.values() for actor in act["actors"]}
    boxes: dict[int, list] = defaultdict(list)
    path = KITWARE_DIR / f"{video_name}.geom.yml"
    if path.exists():
        for item in yaml.safe_load(path.read_text()) or []:
            rec = item.get("geom") if isinstance(item, dict) else None
            if not rec or int(rec["id1"]) not in wanted:
                continue
            frame = int(rec["ts0"])
            if frame % MEVA_BOX_STRIDE and not rec.get("keyframe"):
                continue
            x0, y0, x1, y1 = (int(float(v)) for v in str(rec["g0"]).split())
            boxes[int(rec["id1"])].append([frame, x0, y0, x1, y1])
    actors = {
        str(actor): {"type": types.get(actor, "unknown"), "boxes": sorted(boxes.get(actor, []))}
        for actor in sorted(wanted)
    }
    return activities, actors


def _index_path():
    return OUTPUT_DIR / "index.json"


def _read_index() -> dict:
    try:
        return json.loads(_index_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"videos": []}


def _db_changed_at(engine, video_id: str):
    """Newest change in the database that affects one video's explorer file, or None."""
    row = pd.read_sql(
        text("""
        SELECT GREATEST(
            (SELECT MAX(created_at) FROM bronze.videos WHERE video_id = :id),
            (SELECT MAX(evaluated_at) FROM bronze.meva_event_coverage WHERE video_id = :id),
            (SELECT MAX(ar.evaluated_at) FROM bronze.annotations_report ar
             JOIN bronze.events e ON e.event_id = ar.event_id WHERE e.video_id = :id)
        ) AS changed_at
        """),
        engine,
        params={"id": video_id},
    )
    value = row["changed_at"].iloc[0]
    return None if pd.isna(value) else pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value).tz_localize("UTC")


def build(video_filter: str | None = None, only_if_stale: bool = False) -> list[dict]:
    """Export explorer data for every video, or for the videos matching `video_filter`.

    Returns the index entries that were (re)built. With `only_if_stale` and a filter that
    selects videos already exported, a video whose file is newer than the last change to its
    rows in the database is left as it is; its existing index entry is returned instead.
    """
    engine = make_engine()
    started = time.time()

    def read(sql, label=None, **params):
        if label:
            print(f"  {label}...", end="", flush=True)
        t = time.time()
        frame = pd.read_sql(text(sql), engine, params=params or None)
        if label:
            print(f" {len(frame):,} rows in {time.time() - t:.0f}s", flush=True)
        return frame

    existing_index = _read_index()
    existing_entries = {entry["id"]: entry for entry in existing_index.get("videos", [])}
    fresh: list[dict] = []
    selected_ids = None
    if video_filter:
        # Resolve the filter and check freshness before the heavy reads, so an up-to-date video
        # costs a couple of small queries rather than a scan of the coverage table.
        names = read(
            "SELECT video_id, video_name FROM bronze.videos WHERE video_name LIKE :pattern OR video_id = :exact",
            pattern=f"%{video_filter}%", exact=video_filter,
        )
        if names.empty:
            raise SystemExit(f"No video in bronze.videos matches {video_filter!r}.")
        selected_ids = sorted(names["video_id"])
        if only_if_stale:
            for vid in list(selected_ids):
                entry = existing_entries.get(vid)
                if not entry or not (OUTPUT_DIR / entry["file"]).exists():
                    continue
                # Files from before build stamps existed count as built when they were last written.
                built_at = entry.get("built_at") or datetime.fromtimestamp(
                    (OUTPUT_DIR / entry["file"]).stat().st_mtime, tz=timezone.utc
                ).isoformat()
                changed_at = _db_changed_at(engine, vid)
                if changed_at is not None and pd.Timestamp(built_at) >= changed_at:
                    fresh.append(entry)
                    selected_ids.remove(vid)
            if not selected_ids:
                print(f"{len(fresh)} video(s) already up to date with the database ({time.time() - started:.0f}s).", flush=True)
                return fresh

    print("Reading the bronze schema (the boxes query is the slow one, about a minute):", flush=True)

    coverage = read("""
        SELECT DISTINCT ON (pair_id, judge_model) *
        FROM bronze.meva_event_coverage
        WHERE status IN ('judged', 'no_annotations')
    """ + ("  AND video_id = ANY(:ids)\n" if selected_ids else "") + """
        ORDER BY pair_id, judge_model, evaluated_at DESC
    """, "coverage verdicts", **({"ids": list(selected_ids)} if selected_ids else {}))
    if coverage.empty:
        raise SystemExit(
            f"bronze.meva_event_coverage has no evaluable rows for {video_filter!r}; run notebook 08 for it first."
            if video_filter else "bronze.meva_event_coverage has no evaluable rows; run notebook 08 first."
        )
    # One judge model per export: the newest one seen, so a comparison stays like for like.
    judge_model = coverage.sort_values("evaluated_at")["judge_model"].iloc[-1]
    coverage = coverage[coverage["judge_model"] == judge_model]
    video_ids = sorted(coverage["video_id"].unique())

    videos = read("SELECT * FROM bronze.videos WHERE video_id = ANY(:ids)", "videos", ids=list(video_ids))
    ground_truth = read(
        "SELECT * FROM bronze.ground_truth WHERE video_id = ANY(:ids)", "MEVA ground truth", ids=list(video_ids)
    )
    events = read(
        "SELECT * FROM bronze.events WHERE video_id = ANY(:ids) ORDER BY start_seconds",
        "pipeline events", ids=list(video_ids),
    )
    event_objects = read(
        """SELECT eo.event_id, eo.object_id, o.source_object_id, o.label_details
           FROM bronze.event_objects eo
           JOIN bronze.events e USING (event_id)
           JOIN bronze.objects o USING (object_id)
           WHERE e.video_id = ANY(:ids)""",
        "event objects", ids=list(video_ids),
    )
    # Boxes: filter by the distinct objects first (an object in several events would otherwise be
    # multiplied by the join), and unpack the JSONB on the server so only five integers travel per row.
    geometries = read(
        """SELECT g.object_id, g.frame_index, g.label,
                  (g.bounding_box_pixels->>'x_min')::int AS x0, (g.bounding_box_pixels->>'y_min')::int AS y0,
                  (g.bounding_box_pixels->>'x_max')::int AS x1, (g.bounding_box_pixels->>'y_max')::int AS y1
           FROM bronze.geometries g
           WHERE g.object_id IN (
               SELECT DISTINCT eo.object_id
               FROM bronze.event_objects eo JOIN bronze.events e USING (event_id)
               WHERE e.video_id = ANY(:ids))""",
        "pipeline boxes", ids=list(video_ids),
    )
    # Agreement (Part B): the newest prompt version in the table, one verdict per pipeline event.
    agreement_version = None
    agreement_by_event: dict[str, dict] = {}
    try:
        versions = read("""
            SELECT prompt_version, MAX(evaluated_at) AS newest
            FROM bronze.annotations_report
            WHERE prompt_version IS NOT NULL
            GROUP BY prompt_version ORDER BY newest DESC LIMIT 1
        """)
        if not versions.empty:
            agreement_version = versions["prompt_version"].iloc[0]
            agreement = read(
                """SELECT DISTINCT ON (ar.event_id)
                          ar.event_id, ar.llm_score, ar.llm_match, ar.llm_reason, ar.eval_status, ar.meva_reference
                   FROM bronze.annotations_report ar
                   JOIN bronze.events e ON e.event_id = ar.event_id
                   WHERE e.video_id = ANY(:ids) AND ar.prompt_version = :version
                   ORDER BY ar.event_id, ar.evaluated_at DESC""",
                f"agreement verdicts ({agreement_version})", ids=list(video_ids), version=agreement_version,
            )
            for row in agreement.itertuples(index=False):
                agreement_by_event[row.event_id] = {
                    "score": float(row.llm_score) if pd.notna(row.llm_score) else None,
                    "match": bool(row.llm_match) if pd.notna(row.llm_match) else None,
                    "reason": row.llm_reason,
                    "status": row.eval_status,
                    "reference": row.meva_reference,
                }
    except Exception as error:  # the table is written by notebook 08 Part B and may not exist yet
        print(f"  agreement verdicts unavailable ({type(error).__name__}); events will show as not judged")

    print(f"{len(video_ids)} videos, {len(ground_truth)} MEVA instances, {len(events)} events, "
          f"{len(geometries)} pipeline boxes, judge {judge_model}, agreement {agreement_version or 'none'} "
          f"({time.time() - started:.0f}s so far)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not video_filter:
        for old in OUTPUT_DIR.glob("*.json"):
            old.unlink()

    boxes_by_object: dict[str, list] = defaultdict(list)
    labels_by_object: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for row in geometries.itertuples(index=False):
        boxes_by_object[row.object_id].append(
            [int(row.frame_index), int(row.x0), int(row.y0), int(row.x1), int(row.y1)]
        )
        labels_by_object[row.object_id][row.label or "object"] += 1

    index = []
    for video in videos.sort_values("video_name").itertuples(index=False):
        vid = video.video_id
        activities_yml, actors = read_kitware(video.video_name)
        verdicts = {
            row.activity_type: row
            for row in coverage[coverage["video_id"] == vid].itertuples(index=False)
        }

        meva_activities = []
        for gt in ground_truth[ground_truth["video_id"] == vid].sort_values("start_frame").itertuples(index=False):
            verdict = verdicts.get(gt.activity_type)
            yml = activities_yml.get(int(gt.meva_event_id)) if pd.notna(gt.meva_event_id) else None
            meva_activities.append({
                "gt_id": int(gt.id),
                "meva_event_id": int(gt.meva_event_id) if pd.notna(gt.meva_event_id) else None,
                "type": gt.activity_type,
                "start_frame": int(gt.start_frame),
                "end_frame": int(gt.end_frame),
                "actors": [str(a) for a in (yml["actors"] if yml else [])],
                "covered": bool(verdict.covered) if verdict is not None and pd.notna(verdict.covered) else None,
                "score": float(verdict.coverage_score) if verdict is not None and pd.notna(verdict.coverage_score) else None,
                "reason": verdict.judge_reason if verdict is not None else None,
                "covering_event_id": verdict.covering_event_id if verdict is not None and pd.notna(verdict.covering_event_id) else None,
                "status": verdict.status if verdict is not None else "not_evaluated",
            })

        video_events = events[events["video_id"] == vid]
        objects: dict[str, dict] = {}
        pipeline_events = []
        for ev in video_events.itertuples(index=False):
            links = event_objects[event_objects["event_id"] == ev.event_id]
            involved = []
            for link in links.itertuples(index=False):
                key = str(link.source_object_id)
                involved.append(key)
                if key not in objects:
                    details = link.label_details
                    if isinstance(details, str):
                        details = json.loads(details)
                    counts = labels_by_object.get(link.object_id, {})
                    label = max(counts, key=counts.get) if counts else (details[0]["label"] if details else "object")
                    objects[key] = {
                        "label": label,
                        "descriptions": sorted({d for item in (details or []) for d in item.get("descriptions", [])}),
                        "boxes": sorted(boxes_by_object.get(link.object_id, [])),
                    }
            actions = ev.actions if isinstance(ev.actions, list) else (json.loads(ev.actions) if isinstance(ev.actions, str) else [])
            other = ev.other_actions if isinstance(ev.other_actions, list) else (json.loads(ev.other_actions) if isinstance(ev.other_actions, str) else [])
            pipeline_events.append({
                "id": ev.event_id,
                "name": ev.event_name,
                "description": ev.description,
                "actions": actions or [],
                "other_actions": other or [],
                "start_frame": int(ev.start_frame) if pd.notna(ev.start_frame) else None,
                "end_frame": int(ev.end_frame) if pd.notna(ev.end_frame) else None,
                "start_s": float(ev.start_seconds),
                "end_s": float(ev.end_seconds),
                "objects": sorted(set(involved), key=lambda s: int(s) if s.isdigit() else s),
                "agreement": agreement_by_event.get(ev.event_id),
            })

        local_mp4 = LOCAL_MP4_DIR / f"{vid}.mp4"
        pairs = [v for v in verdicts.values()]
        built_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "built_at": built_at,
            "video": {
                "id": vid,
                "name": video.video_name,
                "camera": video.camera_id,
                "scene": video.scene,
                "captured": video.capture_start_local.isoformat() if pd.notna(video.capture_start_local) else None,
                "url": video.video_url,
                "local_url": str(local_mp4.relative_to(PROJECT_ROOT)) if local_mp4.exists() else None,
                "width": int(video.frame_width),
                "height": int(video.frame_height),
                "fps": float(video.fps),
                "duration": float(video.duration_seconds),
            },
            "judge_model": judge_model,
            "agreement_version": agreement_version,
            "meva": {"activities": meva_activities, "actors": actors},
            "pipeline": {"events": pipeline_events, "objects": objects},
        }
        file_name = f"{vid}.json"
        (OUTPUT_DIR / file_name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        index.append({
            "id": vid,
            "name": video.video_name,
            "camera": video.camera_id,
            "scene": video.scene,
            "captured": payload["video"]["captured"],
            "file": file_name,
            "pairs": len(pairs),
            "covered": int(sum(1 for v in pairs if v.covered is True)),
            "instances": len(meva_activities),
            "events": len(pipeline_events),
            "events_judged": sum(1 for e in pipeline_events if e["agreement"] and e["agreement"]["status"] == "scored"),
            "events_matched": sum(1 for e in pipeline_events if e["agreement"] and e["agreement"]["match"] is True),
            "built_at": built_at,
        })
        print(f"  {video.video_name}: {len(meva_activities)} MEVA instances, {len(pipeline_events)} events, "
              f"{sum(len(a['boxes']) for a in actors.values())} MEVA boxes, "
              f"{sum(len(o['boxes']) for o in objects.values())} pipeline boxes")

    built = list(index)
    if video_filter:
        # Replace the rebuilt videos' entries in the existing index and keep the rest.
        rebuilt_ids = {entry["id"] for entry in index}
        index = sorted(
            [entry for entry in existing_index.get("videos", []) if entry["id"] not in rebuilt_ids] + index,
            key=lambda entry: entry["name"],
        )
    _index_path().write_text(
        json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": judge_model,
            "agreement_version": agreement_version,
            "videos": index,
        }, indent=1),
        encoding="utf-8",
    )
    total = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.json"))
    print(f"Wrote {len(built)} videos to {OUTPUT_DIR} ({total / 1e6:.1f} MB) in {time.time() - started:.0f}s")
    return built + fresh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", help="rebuild only videos whose name contains this text (or this video id)")
    args = parser.parse_args()
    build(args.video)


if __name__ == "__main__":
    main()
