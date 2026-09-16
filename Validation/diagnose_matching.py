# Validation/diagnose_matching.py

import pandas as pd
import re


def norm(x):
    return re.sub(
        r"_+", "_",
        re.sub(r"[^a-z0-9]+", "_", str(x).lower())
    ).strip("_")


events = pd.read_csv("data/processed/test_10/events.csv")
videos = pd.read_csv("data/processed/test_10/videos.csv")
meva = pd.read_csv("data/interim/meva_reference/meva_events.csv")

generated = events.merge(
    videos[["video_id", "video_name"]],
    on="video_id",
    how="left"
)

generated["norm"] = generated["event_name"].apply(norm)
meva["norm"] = meva["activity"].apply(norm)

print("\n=== GENERATED LABELS ===")
for x in sorted(generated["norm"].unique()):
    print(x)

print("\n=== MEVA LABELS ===")
for x in sorted(meva["norm"].unique()):
    print(x)

print("\n=== EXACT LABEL OVERLAP ===")

overlap = (
    set(generated["norm"])
    & set(meva["norm"])
)

for x in sorted(overlap):
    print(x)

print(f"\nExact overlapping labels: {len(overlap)}")

print("\n=== VIDEO CHECK ===")

gen_videos = set(generated["video_name"])
meva_videos = set(meva["video_name"])

print("Generated:", len(gen_videos))
print("MEVA:", len(meva_videos))

print("\nOnly generated:")
print(gen_videos - meva_videos)

print("\nOnly MEVA:")
print(meva_videos - gen_videos)

print("\n=== FRAME RANGE EXAMPLE ===")

for video in sorted(gen_videos & meva_videos)[:3]:

    print(f"\n{video}")

    g = generated[generated.video_name == video]
    m = meva[meva.video_name == video]

    print(
        "Generated:",
        g[["event_name", "start_frame", "end_frame"]]
        .head(5)
        .to_dict("records")
    )

    print(
        "MEVA:",
        m[["activity", "start_frame", "end_frame"]]
        .head(5)
        .to_dict("records")
    )