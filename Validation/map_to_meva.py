from pathlib import Path
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from meva_ontology import MEVA_ACTIVITIES


EVENTS_PATH = Path("data/processed/test_10/events.csv")
OUTPUT_PATH = Path(
    "data/interim/validation/generated_meva_mapping.csv"
)

MODEL_NAME = "all-MiniLM-L6-v2"

# Semantic threshold is only used AFTER an event
# passes the explicit MEVA evidence gate.
MIN_SIMILARITY = 0.45


# ---------------------------------------------------------
# Evidence rules
# ---------------------------------------------------------
# These rules determine whether the generated annotation
# actually contains evidence for a MEVA activity.
#
# If no rule is satisfied, the event becomes NONE.
# ---------------------------------------------------------

EVIDENCE_RULES = {
    "person_exits_vehicle": [
        r"\bperson\b.{0,40}\b(exits|exited|leaves|left|gets out of)\b.{0,30}\b(vehicle|car|truck|suv|van)\b",
    ],

    "person_enters_vehicle": [
        r"\bperson\b.{0,40}\b(enters|entered|gets into)\b.{0,30}\b(vehicle|car|truck|suv|van)\b",
    ],

    "person_opens_vehicle_door": [
        r"\bperson\b.{0,40}\b(opens|opened|opening)\b.{0,20}\b(vehicle|car|truck|suv|van)?\s*door\b",
    ],

    "person_closes_vehicle_door": [
        r"\bperson\b.{0,40}\b(closes|closed|closing)\b.{0,20}\b(vehicle|car|truck|suv|van)?\s*door\b",
    ],

    "person_opens_trunk": [
        r"\bperson\b.{0,40}\b(opens|opened|opening)\b.{0,20}\b(trunk|boot)\b",
    ],

    "person_closes_trunk": [
        r"\bperson\b.{0,40}\b(closes|closed|closing)\b.{0,20}\b(trunk|boot)\b",
    ],

    "person_enters_scene_through_structure": [
        r"\bperson\b.{0,40}\b(enters|entered|entering)\b.{0,30}\b(building|doorway|entrance|facility|structure)\b",
    ],

    "person_exits_scene_through_structure": [
        r"\bperson\b.{0,40}\b(exits|exited|exiting|leaves|left)\b.{0,30}\b(building|doorway|entrance|facility|structure)\b",
    ],

    "person_picks_up_object": [
        r"\bperson\b.{0,50}\b(picks up|picked up|picking up)\b",
    ],

    "person_puts_down_object": [
        r"\bperson\b.{0,50}\b(puts down|put down|putting down)\b",
    ],

    "person_carries_heavy_object": [
        r"\bperson\b.{0,40}\b(carries|carried|carrying)\b.{0,30}\b(object|box|bag|suitcase|package)\b",
    ],

    "person_transfers_object": [
        r"\bperson\b.{0,50}\b(transfers|transferred|hands|handed|gives|gave)\b.{0,30}\b(object|item|bag|box|package)\b",
    ],

    "person_unloads_vehicle": [
        r"\bperson\b.{0,50}\b(unloads|unloaded|unloading|removes|removed|removing)\b.{0,30}\b(vehicle|car|truck|suv|van|trunk|boot)\b",
    ],

    "person_talks_to_person": [
        r"\b(person|people|individuals)\b.{0,50}\b(talks|talking|speaks|speaking|conversation|conversing)\b",
    ],

    "vehicle_starts": [
        r"\b(vehicle|car|truck|suv|van)\b.{0,50}\b(starts moving|started moving|begins moving|began moving)\b",
    ],

    "vehicle_stops": [
        r"\b(vehicle|car|truck|suv|van)\b.{0,80}\b(stops|stopped|comes to a stop|came to a stop)\b",
    ],

    "vehicle_reverses": [
        r"\b(vehicle|car|truck|suv|van)\b.{0,50}\b(reverses|reversed|reversing|backs up|backing up|moves backwards|moving backwards)\b",
    ],

    "vehicle_turns_left": [
        r"\b(vehicle|car|truck|suv|van)\b.{0,50}\b(turns left|turned left|turning left|left turn)\b",
    ],

    "vehicle_turns_right": [
        r"\b(vehicle|car|truck|suv|van)\b.{0,50}\b(turns right|turned right|turning right|right turn)\b",
    ],

    "vehicle_makes_u_turn": [
        r"\b(vehicle|car|truck|suv|van)\b.{0,50}\b(makes a u-turn|making a u-turn|u-turn)\b",
    ],

    "vehicle_drops_off_person": [
        r"\b(vehicle|car|truck|suv|van)\b.{0,50}\b(drops off|dropped off|dropping off)\b.{0,30}\b(person|passenger)\b",
    ],
}

def build_generated_text(row):
    name = str(row["event_name"]).strip()

    description = ""
    if pd.notna(row.get("description")):
        description = str(row["description"]).strip()

    if description:
        return f"{name}. {description}"

    return name


def find_evidence_classes(text):
    """
    Return MEVA classes for which explicit textual
    evidence exists in the generated annotation.
    """

    text = text.lower()

    matches = []

    for activity, patterns in EVIDENCE_RULES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append(activity)
                break

    return matches


def main():

    print("\n=== MEVA Ontology Mapper V2 ===\n")

    events = pd.read_csv(EVENTS_PATH)

    events["generated_text"] = events.apply(
        build_generated_text,
        axis=1,
    )

    print(f"Generated events: {len(events)}")
    print(f"MEVA classes    : {len(MEVA_ACTIVITIES)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    # -------------------------------------------------
    # Encode ontology
    # -------------------------------------------------

    ontology_labels = list(MEVA_ACTIVITIES.keys())

    ontology_texts = [
        f"{label.replace('_', ' ')}. {description}"
        for label, description in MEVA_ACTIVITIES.items()
    ]

    ontology_embeddings = model.encode(
        ontology_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    ontology_index = {
        label: i
        for i, label in enumerate(ontology_labels)
    }

    # -------------------------------------------------
    # Encode generated annotations
    # -------------------------------------------------

    generated_embeddings = model.encode(
        events["generated_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    mappings = []

    # -------------------------------------------------
    # Evidence gate + semantic classification
    # -------------------------------------------------

    for i, row in events.iterrows():

        text = row["generated_text"]

        evidence_classes = find_evidence_classes(text)

        # ---------------------------------------------
        # No explicit MEVA evidence
        # ---------------------------------------------

        if not evidence_classes:

            mappings.append(
                {
                    "event_id": row["event_id"],
                    "event_name": row["event_name"],
                    "description": row.get("description"),
                    "generated_text": text,

                    "mapped_meva_activity": "NONE",
                    "mapping_similarity": 0.0,

                    "evidence_classes": "NONE",
                    "mapping_reason": "no_explicit_meva_evidence",
                }
            )

            continue

        # ---------------------------------------------
        # Compare ONLY against evidence-supported
        # MEVA classes.
        # ---------------------------------------------

        candidates = []

        for activity in evidence_classes:

            idx = ontology_index[activity]

            similarity = float(
                np.dot(
                    ontology_embeddings[idx],
                    generated_embeddings[i],
                )
            )

            candidates.append(
                (
                    activity,
                    similarity,
                )
            )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        best_activity, best_similarity = candidates[0]

        # ---------------------------------------------
        # Final rejection
        # ---------------------------------------------

        if best_similarity >= MIN_SIMILARITY:
            mapped_activity = best_activity
            reason = "explicit_evidence_and_semantic_match"
        else:
            mapped_activity = "NONE"
            reason = "semantic_similarity_below_threshold"

        mappings.append(
            {
                "event_id": row["event_id"],
                "event_name": row["event_name"],
                "description": row.get("description"),
                "generated_text": text,

                "mapped_meva_activity": mapped_activity,
                "mapping_similarity": round(
                    best_similarity,
                    4,
                ),

                "evidence_classes": "|".join(
                    evidence_classes
                ),

                "mapping_reason": reason,
            }
        )

    output = pd.DataFrame(mappings)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    mapped = (
        output["mapped_meva_activity"] != "NONE"
    ).sum()

    none = (
        output["mapped_meva_activity"] == "NONE"
    ).sum()

    print("\n------------------------------")
    print("ONTOLOGY MAPPING SUMMARY")
    print("------------------------------")

    print(f"Total events : {len(output)}")
    print(f"Mapped       : {mapped}")
    print(f"NONE         : {none}")

    print("\nMapped class distribution:\n")

    print(
        output["mapped_meva_activity"]
        .value_counts()
        .to_string()
    )

    print("\nMapping reasons:\n")

    print(
        output["mapping_reason"]
        .value_counts()
        .to_string()
    )

    print(f"\nOutput: {OUTPUT_PATH}")

    print(
        "\n[DONE] Evidence-gated ontology "
        "mapping completed."
    )


if __name__ == "__main__":
    main()