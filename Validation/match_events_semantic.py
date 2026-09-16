from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"

# Candidate must overlap temporally at least this much.
MIN_TEMPORAL_IOU = 0.10

# Semantic similarity below this is considered too weak.
MIN_SEMANTIC_SIMILARITY = 0.35

# Final combined score weights.
SEMANTIC_WEIGHT = 0.60
TEMPORAL_WEIGHT = 0.40


# ---------------------------------------------------------
# TEMPORAL IOU
# ---------------------------------------------------------

def temporal_iou(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:

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


# ---------------------------------------------------------
# TEXT PREPARATION
# ---------------------------------------------------------

def meva_to_text(activity: str) -> str:
    """
    Convert MEVA taxonomy label into natural language.

    person_exits_scene_through_structure
        ->
    person exits scene through structure
    """

    return str(activity).replace("_", " ").strip().lower()


def generated_to_text(row: pd.Series) -> str:
    """
    Combine generated event name and description.

    This gives semantic matching more context than
    event_name alone.
    """

    name = str(row["event_name"]).strip()

    description = ""

    if (
        "description" in row.index
        and pd.notna(row["description"])
    ):
        description = str(row["description"]).strip()

    if description:
        return f"{name}. {description}"

    return name


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data(
    events_path: Path,
    videos_path: Path,
    meva_path: Path,
):

    events = pd.read_csv(events_path)
    videos = pd.read_csv(videos_path)
    meva = pd.read_csv(meva_path)

    generated = events.merge(
        videos[["video_id", "video_name"]],
        on="video_id",
        how="left",
        validate="many_to_one",
    )

    generated["semantic_text"] = generated.apply(
        generated_to_text,
        axis=1,
    )

    meva["semantic_text"] = (
        meva["activity"]
        .apply(meva_to_text)
    )

    return generated, meva, videos


# ---------------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------------

def create_embeddings(
    model: SentenceTransformer,
    generated: pd.DataFrame,
    meva: pd.DataFrame,
):

    generated_text = generated[
        "semantic_text"
    ].tolist()

    meva_text = meva[
        "semantic_text"
    ].tolist()

    generated_embeddings = model.encode(
        generated_text,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    meva_embeddings = model.encode(
        meva_text,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return (
        generated_embeddings,
        meva_embeddings,
    )


# ---------------------------------------------------------
# MATCH ONE VIDEO
# ---------------------------------------------------------

def match_video(
    video_name: str,
    generated_video: pd.DataFrame,
    meva_video: pd.DataFrame,
    generated_embeddings: np.ndarray,
    meva_embeddings: np.ndarray,
):

    results = []

    # No generated annotations -> every MEVA event is FN.
    if generated_video.empty:

        for meva_idx, truth in meva_video.iterrows():

            results.append(
                {
                    "video_name": video_name,
                    "generated_event_id": None,
                    "meva_event_id": truth["meva_event_id"],
                    "generated_activity": None,
                    "generated_description": None,
                    "meva_activity": truth["activity"],
                    "semantic_similarity": 0.0,
                    "temporal_iou": 0.0,
                    "combined_score": 0.0,
                    "match_status": "FN",
                }
            )

        return results

    if meva_video.empty:

        for gen_idx, gen in generated_video.iterrows():

            results.append(
                {
                    "video_name": video_name,
                    "generated_event_id": gen["event_id"],
                    "meva_event_id": None,
                    "generated_activity": gen["event_name"],
                    "generated_description": gen.get(
                        "description",
                        None,
                    ),
                    "meva_activity": None,
                    "semantic_similarity": 0.0,
                    "temporal_iou": 0.0,
                    "combined_score": 0.0,
                    "match_status": "FP",
                }
            )

        return results

    gen_indices = generated_video.index.tolist()
    meva_indices = meva_video.index.tolist()

    score_matrix = np.zeros(
        (
            len(gen_indices),
            len(meva_indices),
        )
    )

    metadata = {}

    # -----------------------------------------------------
    # CREATE CANDIDATE MATRIX
    # -----------------------------------------------------

    for i, gen_idx in enumerate(gen_indices):

        gen = generated_video.loc[gen_idx]

        for j, meva_idx in enumerate(meva_indices):

            truth = meva_video.loc[meva_idx]

            t_iou = temporal_iou(
                int(gen["start_frame"]),
                int(gen["end_frame"]),
                int(truth["start_frame"]),
                int(truth["end_frame"]),
            )

            # No meaningful temporal overlap -> impossible match.
            if t_iou < MIN_TEMPORAL_IOU:
                continue

            semantic_similarity = float(
                np.dot(
                    generated_embeddings[gen_idx],
                    meva_embeddings[meva_idx],
                )
            )

            if (
                semantic_similarity
                < MIN_SEMANTIC_SIMILARITY
            ):
                continue

            combined_score = (
                SEMANTIC_WEIGHT
                * semantic_similarity
                +
                TEMPORAL_WEIGHT
                * t_iou
            )

            score_matrix[i, j] = combined_score

            metadata[(i, j)] = {
                "semantic_similarity":
                    semantic_similarity,
                "temporal_iou":
                    t_iou,
                "combined_score":
                    combined_score,
            }

    # -----------------------------------------------------
    # HUNGARIAN ONE-TO-ONE ASSIGNMENT
    # -----------------------------------------------------

    row_ind, col_ind = linear_sum_assignment(
        -score_matrix
    )

    matched_gen = set()
    matched_meva = set()

    for i, j in zip(row_ind, col_ind):

        if (i, j) not in metadata:
            continue

        gen_idx = gen_indices[i]
        meva_idx = meva_indices[j]

        gen = generated_video.loc[gen_idx]
        truth = meva_video.loc[meva_idx]

        scores = metadata[(i, j)]

        matched_gen.add(gen_idx)
        matched_meva.add(meva_idx)

        results.append(
            {
                "video_name": video_name,
                "generated_event_id":
                    gen["event_id"],
                "meva_event_id":
                    truth["meva_event_id"],
                "generated_activity":
                    gen["event_name"],
                "generated_description":
                    gen.get("description", None),
                "meva_activity":
                    truth["activity"],
                "semantic_similarity":
                    round(
                        scores[
                            "semantic_similarity"
                        ],
                        4,
                    ),
                "temporal_iou":
                    round(
                        scores["temporal_iou"],
                        4,
                    ),
                "combined_score":
                    round(
                        scores["combined_score"],
                        4,
                    ),
                "match_status": "TP",
            }
        )

    # -----------------------------------------------------
    # FALSE POSITIVES
    # -----------------------------------------------------

    for gen_idx, gen in generated_video.iterrows():

        if gen_idx in matched_gen:
            continue

        results.append(
            {
                "video_name": video_name,
                "generated_event_id":
                    gen["event_id"],
                "meva_event_id": None,
                "generated_activity":
                    gen["event_name"],
                "generated_description":
                    gen.get("description", None),
                "meva_activity": None,
                "semantic_similarity": 0.0,
                "temporal_iou": 0.0,
                "combined_score": 0.0,
                "match_status": "FP",
            }
        )

    # -----------------------------------------------------
    # FALSE NEGATIVES
    # -----------------------------------------------------

    for meva_idx, truth in meva_video.iterrows():

        if meva_idx in matched_meva:
            continue

        results.append(
            {
                "video_name": video_name,
                "generated_event_id": None,
                "meva_event_id":
                    truth["meva_event_id"],
                "generated_activity": None,
                "generated_description": None,
                "meva_activity":
                    truth["activity"],
                "semantic_similarity": 0.0,
                "temporal_iou": 0.0,
                "combined_score": 0.0,
                "match_status": "FN",
            }
        )

    return results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

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
        default="data/interim/validation/"
                "semantic_event_matches.csv",
    )

    args = parser.parse_args()

    generated, meva, videos = load_data(
        Path(args.events),
        Path(args.videos),
        Path(args.meva),
    )

    print("\n=== Semantic MEVA Event Matcher ===\n")

    print(
        f"Generated events : {len(generated)}"
    )

    print(
        f"MEVA events      : {len(meva)}"
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    (
        generated_embeddings,
        meva_embeddings,
    ) = create_embeddings(
        model,
        generated,
        meva,
    )

    results = []

    # IMPORTANT:
    # Iterate over videos.csv, NOT generated events.
    # This includes videos with zero generated events.

    for video_name in sorted(
        videos["video_name"].unique()
    ):

        gen_video = generated[
            generated["video_name"]
            == video_name
        ]

        meva_video = meva[
            meva["video_name"]
            == video_name
        ]

        video_results = match_video(
            video_name,
            gen_video,
            meva_video,
            generated_embeddings,
            meva_embeddings,
        )

        results.extend(video_results)

        statuses = pd.Series(
            [
                x["match_status"]
                for x in video_results
            ]
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

    tp = (
        output["match_status"] == "TP"
    ).sum()

    fp = (
        output["match_status"] == "FP"
    ).sum()

    fn = (
        output["match_status"] == "FN"
    ).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    matched = output[
        output["match_status"] == "TP"
    ]

    print("\n------------------------------")
    print("SEMANTIC MATCH SUMMARY")
    print("------------------------------")

    print(f"TP        : {tp}")
    print(f"FP        : {fp}")
    print(f"FN        : {fn}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    if not matched.empty:

        print(
            "Mean semantic similarity : "
            f"{matched['semantic_similarity'].mean():.4f}"
        )

        print(
            "Mean temporal IoU        : "
            f"{matched['temporal_iou'].mean():.4f}"
        )

        print(
            "Mean combined score      : "
            f"{matched['combined_score'].mean():.4f}"
        )

    print(f"\nOutput: {output_path}")

    print(
        "\n[DONE] Semantic matching completed."
    )


if __name__ == "__main__":
    main()