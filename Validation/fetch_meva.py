from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


MEVA_REPO_URL = "https://gitlab.kitware.com/meva/meva-data-repo.git"
MEVA_ANNOTATION_ROOT = Path("annotation/DIVA-phase-2/MEVA/kitware")

ANNOTATION_SUFFIXES = (
    "activities.yml",
    "types.yml",
    "geom.yml",
)


def run_git(args: list[str], cwd: Path | None = None) -> None:
    """Run a Git command and fail immediately if it does not succeed."""
    command = ["git", *args]

    print(f"[git] {' '.join(command)}")

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def load_video_names(videos_csv: Path) -> list[str]:
    """Read unique MEVA video names from videos.csv."""
    df = pd.read_csv(videos_csv)

    if "video_name" not in df.columns:
        raise ValueError(
            f"'video_name' column not found in {videos_csv}"
        )

    names = (
        df["video_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    if not names:
        raise ValueError("No video names found in videos.csv")

    return names


def derive_meva_directory(video_name: str) -> Path:
    """
    Convert:

    2018-03-05.13-20-00.13-25-00.school.G423

    into:

    annotation/DIVA-phase-2/MEVA/kitware/2018-03-05/13
    """
    parts = video_name.split(".")

    if len(parts) < 2:
        raise ValueError(
            f"Cannot derive MEVA path from video name: {video_name}"
        )

    date = parts[0]

    start_time = parts[1]

    if len(start_time) < 2:
        raise ValueError(
            f"Cannot derive hour from video name: {video_name}"
        )

    hour = start_time[:2]

    return MEVA_ANNOTATION_ROOT / date / hour


def prepare_sparse_repo(
    repo_dir: Path,
    required_directories: list[Path],
) -> None:
    """
    Create/update a lightweight MEVA repository and checkout
    only the required annotation directories.
    """

    if not (repo_dir / ".git").exists():
        print("\n[INFO] Creating lightweight MEVA repository...")

        repo_dir.mkdir(parents=True, exist_ok=True)

        run_git(
            [
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                MEVA_REPO_URL,
                str(repo_dir),
            ]
        )

    print("\n[INFO] Configuring sparse checkout...")

    run_git(
        ["sparse-checkout", "init", "--no-cone"],
        cwd=repo_dir,
    )

    sparse_paths = [
        path.as_posix() + "/*"
        for path in required_directories
    ]

    run_git(
        ["sparse-checkout", "set", *sparse_paths],
        cwd=repo_dir,
    )

    print("\n[INFO] Fetching latest MEVA metadata...")

    run_git(
        ["fetch", "--depth=1", "origin", "master"],
        cwd=repo_dir,
    )

    run_git(
        ["checkout", "master"],
        cwd=repo_dir,
    )


def copy_annotations(
    video_names: list[str],
    repo_dir: Path,
    output_dir: Path,
) -> None:
    """Copy only annotation files belonging to requested videos."""

    output_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    missing = []

    print("\n[INFO] Collecting MEVA annotations...\n")

    for video_name in video_names:

        source_directory = (
            repo_dir / derive_meva_directory(video_name)
        )

        print(f"Video: {video_name}")

        for suffix in ANNOTATION_SUFFIXES:

            filename = f"{video_name}.{suffix}"

            source = source_directory / filename
            destination = output_dir / filename

            if source.exists():

                shutil.copy2(source, destination)

                print(f"  [FOUND] {suffix}")
                found += 1

            else:

                print(f"  [MISSING] {suffix}")

                missing.append(
                    {
                        "video_name": video_name,
                        "annotation": suffix,
                        "expected_path": str(source),
                    }
                )

    print("\n------------------------------")
    print("MEVA FETCH SUMMARY")
    print("------------------------------")
    print(f"Videos requested : {len(video_names)}")
    print(f"Files found      : {found}")
    print(f"Files expected   : {len(video_names) * 3}")
    print(f"Files missing    : {len(missing)}")

    if missing:

        missing_df = pd.DataFrame(missing)

        report_path = output_dir / "missing_annotations.csv"

        missing_df.to_csv(
            report_path,
            index=False,
        )

        print(f"\nMissing-file report: {report_path}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Fetch MEVA reference annotations for generated videos."
    )

    parser.add_argument(
        "--videos",
        default="data/processed/test_10/videos.csv",
        help="Path to generated videos.csv",
    )

    parser.add_argument(
        "--output",
        default="data/raw/annotations/meva_original_annotations",
        help="Destination for MEVA reference annotations",
    )

    parser.add_argument(
        "--repo-cache",
        default="data/raw/annotations/kitware/meva_sparse",
        help="Temporary sparse MEVA Git repository",
    )

    args = parser.parse_args()

    videos_csv = Path(args.videos)
    output_dir = Path(args.output)
    repo_dir = Path(args.repo_cache)

    print("\n=== MEVA Annotation Fetcher ===\n")

    video_names = load_video_names(videos_csv)

    print(f"Found {len(video_names)} unique videos.")

    required_directories = sorted(
        {
            derive_meva_directory(name)
            for name in video_names
        }
    )

    print(
        f"Only {len(required_directories)} MEVA "
        "date/hour directories are required:"
    )

    for directory in required_directories:
        print(f"  {directory}")

    prepare_sparse_repo(
        repo_dir,
        required_directories,
    )

    copy_annotations(
        video_names,
        repo_dir,
        output_dir,
    )

    print("\n[DONE] MEVA annotation fetch completed.")


if __name__ == "__main__":
    main()