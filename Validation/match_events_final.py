from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


MAPPING_PATH = Path(
    "data/interim/validation/generated_meva_mapping.csv"
)

EVENTS_PATH = Path(
    "data/processed/test_10/events.csv"
)

VIDEOS_PATH = Path(
    "data/processed/test_10/videos.csv"
)

MEVA_PATH = Path(
    "data/interim/meva_reference/meva_events.csv"
)

OUTPUT_PATH = Path(
    "data/interim/validation/final_event_matches.csv"
)

MIN_IOU = 0.30


def temporal_iou(start_a, end_a, start_b, end_b):
    """
    Inclusive frame-based temporal IoU.
    """

    intersection = max(
        0,
        min(end_a, end_b) - max(start_a, start_b) + 1
    )

    duration_a = end_a - start_a + 1
    duration_b = end_b - start_b + 1

    union = duration_a + duration_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def main():

    print("\n=== FINAL EVENT VALIDATION ===\n")

    mappings = pd.read_csv(MAPPING_PATH)
    events = pd.read_csv(EVENTS_PATH)
    videos = pd.read_csv(VIDEOS_PATH)
    meva = pd.read_csv(MEVA_PATH)

    # --------------------------------------------------
    # Add original generated timing information
    # --------------------------------------------------

    generated = mappings.merge(
        events[
            [
                "event_id",
                "video_id",
                "start_frame",
                "end_frame",
            ]
        ],
        on="event_id",
        how="left",
    )

    # --------------------------------------------------
    # Resolve generated video_id -> video_name
    # --------------------------------------------------

    generated = generated.merge(
        videos[
            [
                "video_id",
                "video_name",
            ]
        ],
        on="video_id",
        how="left",
    )

    print(f"Generated events : {len(generated)}")
    print(f"Mapped events    : {(generated['mapped_meva_activity'] != 'NONE').sum()}")
    print(f"MEVA events      : {len(meva)}")
    print(f"Videos           : {len(videos)}")

    results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    matched_ious = []

    # --------------------------------------------------
    # IMPORTANT:
    # Iterate over ALL videos, including videos where
    # the generator produced zero events.
    # --------------------------------------------------

    all_video_names = sorted(
        set(videos["video_name"].dropna())
        | set(meva["video_name"].dropna())
    )

    for video_name in all_video_names:

        gen_video = generated[
            generated["video_name"] == video_name
        ].copy()

        meva_video = meva[
            meva["video_name"] == video_name
        ].copy()

        # Only mapped generated events can become TP.
        mapped_gen = gen_video[
            gen_video["mapped_meva_activity"] != "NONE"
        ].copy()

        candidate_pairs = []

        # --------------------------------------------------
        # Build valid candidate pairs
        # --------------------------------------------------

        for gen_idx, gen_row in mapped_gen.iterrows():

            for meva_idx, meva_row in meva_video.iterrows():

                # Activity agreement is mandatory.
                if (
                    gen_row["mapped_meva_activity"]
                    != meva_row["activity"]
                ):
                    continue

                iou = temporal_iou(
                    int(gen_row["start_frame"]),
                    int(gen_row["end_frame"]),
                    int(meva_row["start_frame"]),
                    int(meva_row["end_frame"]),
                )

                # Temporal agreement is mandatory.
                if iou < MIN_IOU:
                    continue

                candidate_pairs.append(
                    (
                        gen_idx,
                        meva_idx,
                        iou,
                    )
                )

        # --------------------------------------------------
        # Hungarian assignment
        # --------------------------------------------------

        matched_gen = set()
        matched_meva = set()

        if candidate_pairs:

            gen_indices = sorted(
                set(x[0] for x in candidate_pairs)
            )

            meva_indices = sorted(
                set(x[1] for x in candidate_pairs)
            )

            gen_pos = {
                idx: pos
                for pos, idx in enumerate(gen_indices)
            }

            meva_pos = {
                idx: pos
                for pos, idx in enumerate(meva_indices)
            }

            # Invalid pairs get a very large cost.
            cost_matrix = np.full(
                (
                    len(gen_indices),
                    len(meva_indices),
                ),
                1e6,
                dtype=float,
            )

            iou_lookup = {}

            for gen_idx, meva_idx, iou in candidate_pairs:

                r = gen_pos[gen_idx]
                c = meva_pos[meva_idx]

                # Hungarian minimizes cost.
                cost_matrix[r, c] = 1.0 - iou

                iou_lookup[
                    (gen_idx, meva_idx)
                ] = iou

            rows, cols = linear_sum_assignment(
                cost_matrix
            )

            for r, c in zip(rows, cols):

                # Ignore invalid assignments.
                if cost_matrix[r, c] >= 1e6:
                    continue

                gen_idx = gen_indices[r]
                meva_idx = meva_indices[c]

                iou = iou_lookup[
                    (gen_idx, meva_idx)
                ]

                gen_row = generated.loc[gen_idx]
                meva_row = meva.loc[meva_idx]

                matched_gen.add(gen_idx)
                matched_meva.add(meva_idx)

                matched_ious.append(iou)

                results.append(
                    {
                        "video_name": video_name,

                        "generated_event_id":
                            gen_row["event_id"],

                        "meva_event_id":
                            meva_row["meva_event_id"],

                        "generated_event_name":
                            gen_row["event_name"],

                        "mapped_meva_activity":
                            gen_row[
                                "mapped_meva_activity"
                            ],

                        "meva_activity":
                            meva_row["activity"],

                        "generated_start_frame":
                            gen_row["start_frame"],

                        "generated_end_frame":
                            gen_row["end_frame"],

                        "meva_start_frame":
                            meva_row["start_frame"],

                        "meva_end_frame":
                            meva_row["end_frame"],

                        "temporal_iou":
                            round(iou, 4),

                        "match_status": "TP",
                    }
                )

        # --------------------------------------------------
        # Generated events not matched = FP
        #
        # This includes NONE events because the system
        # generated an event for which no valid MEVA
        # activity match was established.
        # --------------------------------------------------

        for gen_idx, gen_row in gen_video.iterrows():

            if gen_idx in matched_gen:
                continue

            results.append(
                {
                    "video_name": video_name,

                    "generated_event_id":
                        gen_row["event_id"],

                    "meva_event_id": None,

                    "generated_event_name":
                        gen_row["event_name"],

                    "mapped_meva_activity":
                        gen_row[
                            "mapped_meva_activity"
                        ],

                    "meva_activity": None,

                    "generated_start_frame":
                        gen_row["start_frame"],

                    "generated_end_frame":
                        gen_row["end_frame"],

                    "meva_start_frame": None,
                    "meva_end_frame": None,

                    "temporal_iou": 0.0,

                    "match_status": "FP",
                }
            )

        # --------------------------------------------------
        # Ground-truth MEVA events not matched = FN
        # --------------------------------------------------

        for meva_idx, meva_row in meva_video.iterrows():

            if meva_idx in matched_meva:
                continue

            results.append(
                {
                    "video_name": video_name,

                    "generated_event_id": None,

                    "meva_event_id":
                        meva_row["meva_event_id"],

                    "generated_event_name": None,

                    "mapped_meva_activity": None,

                    "meva_activity":
                        meva_row["activity"],

                    "generated_start_frame": None,
                    "generated_end_frame": None,

                    "meva_start_frame":
                        meva_row["start_frame"],

                    "meva_end_frame":
                        meva_row["end_frame"],

                    "temporal_iou": 0.0,

                    "match_status": "FN",
                }
            )

        video_tp = len(matched_gen)
        video_fp = len(gen_video) - video_tp
        video_fn = len(meva_video) - len(matched_meva)

        total_tp += video_tp
        total_fp += video_fp
        total_fn += video_fn

        print(
            f"{video_name}: "
            f"TP={video_tp}, "
            f"FP={video_fp}, "
            f"FN={video_fn}"
        )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_iou = (
        sum(matched_ious) / len(matched_ious)
        if matched_ious
        else 0.0
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n------------------------------")
    print("FINAL VALIDATION SUMMARY")
    print("------------------------------")

    print(f"TP               : {total_tp}")
    print(f"FP               : {total_fp}")
    print(f"FN               : {total_fn}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1               : {f1:.4f}")
    print(f"Mean temporal IoU: {mean_iou:.4f}")

    print(f"\nOutput: {OUTPUT_PATH}")

    print("\n[DONE] Final event validation completed.")


if __name__ == "__main__":
    main()