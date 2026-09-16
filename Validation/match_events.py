from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


IOU_THRESHOLD = 0.50


def normalize_activity(value: str) -> str:
    """
    Convert generated labels into MEVA-style labels.

    Example:
        "Person exits vehicle"
        -> "person_exits_vehicle"
    """
    value = str(value).strip().lower()

    # Convert punctuation / whitespace into underscores.
    value = re.sub(r"[^a-z0-9]+", "_", value)

    # Remove duplicate / leading / trailing underscores.
    value = re.sub(r"_+", "_", value).strip("_")

    return value


def temporal_iou(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:
    """Calculate temporal Intersection over Union."""

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)

    intersection = max(
        0,
        intersection_end - intersection_start + 1,
    )

    duration_a = end_a - start_a + 1
    duration_b = end_b - start_b + 1

    union = duration_a + duration_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_data(
    events_path: Path,
    videos_path: Path,
    meva_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    events = pd.read_csv(events_path)
    videos = pd.read_csv(videos_path)
    meva = pd.read_csv(meva_path)

    required_event_columns = {
        "event_id",
        "video_id",
        "event_name",
        "start_frame",
        "end_frame",
    }

    required_video_columns = {
        "video_id",
        "video_name",
    }

    required_meva_columns = {
        "video_name",
        "meva_event_id",
        "activity",
        "start_frame",
        "end_frame",
    }

    if not required_event_columns.issubset(events.columns):
        raise ValueError(
            f"events.csv missing columns: "
            f"{required_event_columns - set(events.columns)}"
        )

    if not required_video_columns.issubset(videos.columns):
        raise ValueError(
            f"videos.csv missing columns: "
            f"{required_video_columns - set(videos.columns)}"
        )

    if not required_meva_columns.issubset(meva.columns):
        raise ValueError(
            f"meva_events.csv missing columns: "
            f"{required_meva_columns - set(meva.columns)}"
        )

    # Resolve generated video_id -> MEVA-compatible video_name.
    generated = events.merge(
        videos[["video_id", "video_name"]],
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    if generated["video_name"].isna().any():
        missing = generated.loc[
            generated["video_name"].isna(),
            "video_id",
        ].unique()

        raise ValueError(
            f"Some event video_ids were not found in videos.csv: {missing}"
        )

    generated["normalized_activity"] = (
        generated["event_name"]
        .apply(normalize_activity)
    )

    meva["normalized_activity"] = (
        meva["activity"]
        .apply(normalize_activity)
    )

    return generated, meva


def generate_candidates(
    generated_video: pd.DataFrame,
    meva_video: pd.DataFrame,
) -> list[dict]:
    """
    Generate all candidate matches satisfying:
        same normalized activity
        temporal IoU >= threshold
    """

    candidates = []

    for gen_idx, gen in generated_video.iterrows():

        for meva_idx, truth in meva_video.iterrows():

            if (
                gen["normalized_activity"]
                != truth["normalized_activity"]
            ):
                continue

            iou = temporal_iou(
                int(gen["start_frame"]),
                int(gen["end_frame"]),
                int(truth["start_frame"]),
                int(truth["end_frame"]),
            )

            if iou < IOU_THRESHOLD:
                continue

            candidates.append(
                {
                    "gen_idx": gen_idx,
                    "meva_idx": meva_idx,
                    "iou": iou,
                }
            )

    return candidates


def greedy_match(candidates: list[dict]) -> list[dict]:
    """
    One-to-one matching.

    Highest temporal IoU candidates are assigned first.
    """

    candidates = sorted(
        candidates,
        key=lambda x: x["iou"],
        reverse=True,
    )

    used_generated = set()
    used_meva = set()

    matches = []

    for candidate in candidates:

        gen_idx = candidate["gen_idx"]
        meva_idx = candidate["meva_idx"]

        if gen_idx in used_generated:
            continue

        if meva_idx in used_meva:
            continue

        used_generated.add(gen_idx)
        used_meva.add(meva_idx)

        matches.append(candidate)

    return matches


def match_video(
    video_name: str,
    generated_video: pd.DataFrame,
    meva_video: pd.DataFrame,
) -> list[dict]:

    candidates = generate_candidates(
        generated_video,
        meva_video,
    )

    matches = greedy_match(candidates)

    matched_gen = {
        match["gen_idx"]
        for match in matches
    }

    matched_meva = {
        match["meva_idx"]
        for match in matches
    }

    output = []

    # True positives
    for match in matches:

        gen = generated_video.loc[match["gen_idx"]]
        truth = meva_video.loc[match["meva_idx"]]

        output.append(
            {
                "video_name": video_name,
                "generated_event_id": gen["event_id"],
                "meva_event_id": truth["meva_event_id"],
                "generated_activity": gen["event_name"],
                "meva_activity": truth["activity"],
                "generated_start_frame": int(gen["start_frame"]),
                "generated_end_frame": int(gen["end_frame"]),
                "meva_start_frame": int(truth["start_frame"]),
                "meva_end_frame": int(truth["end_frame"]),
                "temporal_iou": round(match["iou"], 4),
                "label_match": True,
                "match_status": "TP",
            }
        )

    # False positives
    for gen_idx, gen in generated_video.iterrows():

        if gen_idx in matched_gen:
            continue

        output.append(
            {
                "video_name": video_name,
                "generated_event_id": gen["event_id"],
                "meva_event_id": None,
                "generated_activity": gen["event_name"],
                "meva_activity": None,
                "generated_start_frame": int(gen["start_frame"]),
                "generated_end_frame": int(gen["end_frame"]),
                "meva_start_frame": None,
                "meva_end_frame": None,
                "temporal_iou": 0.0,
                "label_match": False,
                "match_status": "FP",
            }
        )

    # False negatives
    for meva_idx, truth in meva_video.iterrows():

        if meva_idx in matched_meva:
            continue

        output.append(
            {
                "video_name": video_name,
                "generated_event_id": None,
                "meva_event_id": truth["meva_event_id"],
                "generated_activity": None,
                "meva_activity": truth["activity"],
                "generated_start_frame": None,
                "generated_end_frame": None,
                "meva_start_frame": int(truth["start_frame"]),
                "meva_end_frame": int(truth["end_frame"]),
                "temporal_iou": 0.0,
                "label_match": False,
                "match_status": "FN",
            }
        )

    return output


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Match generated events against MEVA ground truth."
    )

    parser.add_argument(
        "--events",
        default="data/processed/test_10/events.csv",
    )

    parser.add_argument(
        "--videos",
        default="data/processed/test_10/videos.csv",
    )

    parser.add_argument(
        "--meva",
        default="data/interim/meva_reference/meva_events.csv",
    )

    parser.add_argument(
        "--output",
        default="data/interim/validation/event_matches.csv",
    )

    args = parser.parse_args()

    generated, meva = load_data(
        Path(args.events),
        Path(args.videos),
        Path(args.meva),
    )

    generated_videos = set(generated["video_name"])
    meva_videos = set(meva["video_name"])

    missing_meva = generated_videos - meva_videos

    if missing_meva:
        raise ValueError(
            f"Generated videos missing from MEVA reference: "
            f"{sorted(missing_meva)}"
        )

    results = []

    print("\n=== MEVA Event Matcher ===\n")

    for video_name in sorted(generated_videos):

        gen_video = generated[
            generated["video_name"] == video_name
        ]

        meva_video = meva[
            meva["video_name"] == video_name
        ]

        video_results = match_video(
            video_name,
            gen_video,
            meva_video,
        )

        results.extend(video_results)

        statuses = pd.Series(
            [r["match_status"] for r in video_results]
        ).value_counts()

        print(
            f"{video_name}\n"
            f"  Generated: {len(gen_video)} | "
            f"MEVA: {len(meva_video)} | "
            f"TP: {statuses.get('TP', 0)} | "
            f"FP: {statuses.get('FP', 0)} | "
            f"FN: {statuses.get('FN', 0)}"
        )

    output = pd.DataFrame(results)

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    tp = (output["match_status"] == "TP").sum()
    fp = (output["match_status"] == "FP").sum()
    fn = (output["match_status"] == "FN").sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    matched = output[
        output["match_status"] == "TP"
    ]

    mean_iou = (
        matched["temporal_iou"].mean()
        if not matched.empty
        else 0
    )

    print("\n------------------------------")
    print("BASELINE MATCH SUMMARY")
    print("------------------------------")
    print(f"Generated events : {len(generated)}")
    print(f"MEVA events      : {len(meva)}")
    print(f"TP               : {tp}")
    print(f"FP               : {fp}")
    print(f"FN               : {fn}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1               : {f1:.4f}")
    print(f"Mean temporal IoU: {mean_iou:.4f}")
    print(f"\nOutput: {output_path}")

    print("\n[DONE] Baseline event matching completed.")


if __name__ == "__main__":
    main()