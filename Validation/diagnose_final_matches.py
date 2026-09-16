from pathlib import Path

import pandas as pd


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


def temporal_iou(a_start, a_end, b_start, b_end):

    intersection = max(
        0,
        min(a_end, b_end) - max(a_start, b_start) + 1
    )

    a_duration = a_end - a_start + 1
    b_duration = b_end - b_start + 1

    union = a_duration + b_duration - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def main():

    mappings = pd.read_csv(MAPPING_PATH)
    events = pd.read_csv(EVENTS_PATH)
    videos = pd.read_csv(VIDEOS_PATH)
    meva = pd.read_csv(MEVA_PATH)

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

    generated = generated[
        generated["mapped_meva_activity"] != "NONE"
    ]

    print("\n=== FINAL MATCH DIAGNOSTIC ===\n")

    for _, gen in generated.iterrows():

        print("=" * 80)

        print(f"Generated : {gen['event_name']}")
        print(f"Mapped    : {gen['mapped_meva_activity']}")
        print(
            f"Frames    : "
            f"{int(gen['start_frame'])}-"
            f"{int(gen['end_frame'])}"
        )

        print(f"Video     : {gen['video_name']}")

        candidates = meva[
            (meva["video_name"] == gen["video_name"])
            &
            (
                meva["activity"]
                == gen["mapped_meva_activity"]
            )
        ].copy()

        if candidates.empty:

            print(
                "\nNo MEVA event of this activity "
                "exists in this video."
            )

            continue

        candidate_results = []

        for _, gt in candidates.iterrows():

            iou = temporal_iou(
                int(gen["start_frame"]),
                int(gen["end_frame"]),
                int(gt["start_frame"]),
                int(gt["end_frame"]),
            )

            candidate_results.append(
                {
                    "meva_event_id":
                        gt["meva_event_id"],

                    "activity":
                        gt["activity"],

                    "meva_start":
                        int(gt["start_frame"]),

                    "meva_end":
                        int(gt["end_frame"]),

                    "iou":
                        round(iou, 4),
                }
            )

        result = pd.DataFrame(
            candidate_results
        ).sort_values(
            "iou",
            ascending=False,
        )

        print("\nSame-class MEVA candidates:\n")

        print(
            result.to_string(
                index=False
            )
        )

        best_iou = result.iloc[0]["iou"]

        print(
            f"\nBest IoU: {best_iou:.4f}"
        )

        if best_iou >= 0.50:
            print("Result  : SHOULD MATCH")
        elif best_iou > 0:
            print("Result  : TEMPORAL MISS")
        else:
            print("Result  : NO TEMPORAL OVERLAP")

    print("\n[DONE]")


if __name__ == "__main__":
    main()