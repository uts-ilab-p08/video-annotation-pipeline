from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def get_video_name(path: Path) -> str:
    """Remove the .activities.yml suffix from the filename."""
    suffix = ".activities.yml"

    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected activities filename: {path.name}")

    return path.name[:-len(suffix)]


def extract_activity_label(activity_record: dict) -> str | None:
    """
    MEVA KPF activity records contain an 'act' section.
    The activity label is stored as a key inside act['act2'].
    """
    act = activity_record.get("act")

    if not isinstance(act, dict):
        return None

    act2 = act.get("act2")

    if not isinstance(act2, dict) or not act2:
        return None

    return next(iter(act2.keys()))


def extract_timespan(activity_record: dict) -> tuple[int | None, int | None]:
    """
    Extract activity-level frame interval.

    Actual MEVA structure:
        timespan:
          - tsr0: [start_frame, end_frame]
    """

    act = activity_record.get("act")

    if not isinstance(act, dict):
        return None, None

    timespan = act.get("timespan")

    if not isinstance(timespan, list):
        return None, None

    for item in timespan:
        if not isinstance(item, dict):
            continue

        tsr0 = item.get("tsr0")

        if (
            isinstance(tsr0, list)
            and len(tsr0) >= 2
        ):
            return int(tsr0[0]), int(tsr0[1])

    return None, None

def extract_event_id(activity_record: dict) -> str | None:
    """Extract the MEVA activity ID."""
    act = activity_record.get("act")

    if not isinstance(act, dict):
        return None

    event_id = act.get("id2")

    if event_id is None:
        return None

    return str(event_id)


def parse_activity_file(path: Path) -> list[dict]:
    """Parse one MEVA .activities.yml file."""

    video_name = get_video_name(path)

    with path.open("r", encoding="utf-8") as f:
        records = yaml.safe_load(f)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected YAML list in {path}, got {type(records).__name__}"
        )

    parsed = []

    for record in records:

        if not isinstance(record, dict):
            continue

        # Ignore KPF metadata/header records.
        if "act" not in record:
            continue

        activity = extract_activity_label(record)
        start_frame, end_frame = extract_timespan(record)
        event_id = extract_event_id(record)

        if activity is None:
            print(
                f"[WARNING] Activity label missing in "
                f"{path.name}"
            )
            continue

        if start_frame is None or end_frame is None:
            print(
                f"[WARNING] Timespan missing for "
                f"{video_name} / {event_id}"
            )
            continue

        parsed.append(
            {
                "video_name": video_name,
                "meva_event_id": event_id,
                "activity": activity,
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )

    return parsed


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Parse MEVA KPF activity annotations."
    )

    parser.add_argument(
        "--input",
        default="data/raw/annotations/meva_original_annotations",
        help="Folder containing downloaded MEVA annotation files",
    )

    parser.add_argument(
        "--output",
        default="data/interim/meva_reference/meva_events.csv",
        help="Output normalized MEVA events CSV",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    print("\n=== MEVA Activity Parser ===\n")

    activity_files = sorted(
        input_dir.glob("*.activities.yml")
    )

    if not activity_files:
        raise FileNotFoundError(
            f"No .activities.yml files found in {input_dir}"
        )

    print(f"Found {len(activity_files)} activity files.\n")

    all_events = []

    for path in activity_files:

        events = parse_activity_file(path)

        all_events.extend(events)

        print(
            f"[PARSED] {path.name}: "
            f"{len(events)} activities"
        )

    if not all_events:
        raise RuntimeError(
            "No MEVA activities were successfully parsed."
        )

    df = pd.DataFrame(all_events)

    df = df.sort_values(
        ["video_name", "start_frame", "end_frame"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print("\n------------------------------")
    print("MEVA PARSE SUMMARY")
    print("------------------------------")
    print(f"Activity files : {len(activity_files)}")
    print(f"Total events   : {len(df)}")
    print(f"Unique videos  : {df['video_name'].nunique()}")
    print(f"Activity types : {df['activity'].nunique()}")
    print(f"\nOutput: {output_path}")

    print("\nActivity distribution:")
    print(
        df["activity"]
        .value_counts()
        .to_string()
    )

    print("\n[DONE] MEVA activities parsed.")


if __name__ == "__main__":
    main()