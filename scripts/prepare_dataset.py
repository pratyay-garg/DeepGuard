from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import cv2

from src.preprocessing.face_cropper import FaceCropper
from src.preprocessing.face_detector import OpenCVFaceDetector, FaceDetector
from src.preprocessing.video_sampler import VideoSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Celeb-DF v2 dataset for DeepGuard.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/raw/celeb-df-v2",
        help="Root directory of downloaded Celeb-DF v2 videos. "
             "Expected subfolders: Celeb-real, YouTube-real, Celeb-synthesis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Where to write crops and manifests.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=10,
        help="Sampling rate (frames per second).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max-real-videos",
        type=int,
        default=None,
        help="Max number of REAL videos to use (randomly sampled). None = use all.",
    )
    parser.add_argument(
        "--max-fake-videos",
        type=int,
        default=None,
        help="Max number of FAKE videos to use (randomly sampled). None = use all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible video sampling.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that have already been processed.",
    )
    return parser.parse_args()


def split_name_for_video(video_id: str, train_ratio: float, val_ratio: float, test_ratio: float) -> str:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    digest = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)

    if value < train_ratio:
        return "train"
    elif value < train_ratio + val_ratio:
        return "val"
    else:
        return "test"


def is_fake_video(video_path: Path) -> bool:
    """
    Celeb-DF v2 folder structure:
    - Celeb-real/       -> real (590 videos)
    - YouTube-real/     -> real (300 videos)
    - Celeb-synthesis/  -> fake (5,639 videos)
    """
    rel = video_path.as_posix().lower()
    return "celeb-synthesis" in rel or "synthesis" in rel


def discover_celebdf_videos(dataset_root: Path, max_real: int | None, max_fake: int | None, seed: int):
    """
    Discover all Celeb-DF v2 videos with optional subsampling.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Celeb-DF v2 dataset root not found: {dataset_root}")

    real_videos = []
    fake_videos = []

    # Expected subfolders
    real_folders = ["Celeb-real", "YouTube-real"]
    fake_folders = ["Celeb-synthesis"]

    for folder_name in real_folders:
        folder_path = dataset_root / folder_name
        if folder_path.exists():
            for video_path in sorted(folder_path.rglob("*.mp4")):
                real_videos.append(video_path)
            

    for folder_name in fake_folders:
        folder_path = dataset_root / folder_name
        if folder_path.exists():
            for video_path in sorted(folder_path.rglob("*.mp4")):
                fake_videos.append(video_path)


    print(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos.")

    # Optional subsampling
    random.seed(seed)
    if max_real is not None and max_real < len(real_videos):
        real_videos = random.sample(real_videos, max_real)
        print(f"  -> Sampled {max_real} real videos (seed={seed})")
    if max_fake is not None and max_fake < len(fake_videos):
        fake_videos = random.sample(fake_videos, max_fake)
        print(f"  -> Sampled {max_fake} fake videos (seed={seed})")

    all_videos = real_videos + fake_videos
    if not all_videos:
        raise FileNotFoundError(
            f"No Celeb-DF v2 videos found under {dataset_root}. "
            "Expected folders: Celeb-real, YouTube-real, Celeb-synthesis"
        )

    return all_videos


def build_video_id(video_path: Path, dataset_root: Path) -> str:
    rel = video_path.relative_to(dataset_root)
    return rel.with_suffix("").as_posix().replace("/", "__")


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    detector = OpenCVFaceDetector(conf_threshold=0.6)
    cropper = FaceCropper(target_size=(224, 224))

    videos = discover_celebdf_videos(
        dataset_root,
        max_real=args.max_real_videos,
        max_fake=args.max_fake_videos,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {"train": [], "val": [], "test": []}
    stats = {"processed": 0, "skipped": 0, "failed": 0, "total_faces": 0}

    for video_path in videos:
        label = 1 if is_fake_video(video_path) else 0
        video_id = build_video_id(video_path, dataset_root)
        split = split_name_for_video(video_id, args.train_ratio, args.val_ratio, args.test_ratio)

        save_dir = output_dir / "faces" / split / ("fake" if label == 1 else "real") / video_id

        # Skip if already processed and --skip-existing is set
        if args.skip_existing and save_dir.exists() and any(save_dir.iterdir()):
            print(f"[SKIP] {video_id}: already processed")
            stats["skipped"] += 1
            continue

        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            sampler = VideoSampler(str(video_path), target_fps=args.target_fps)
        except Exception as e:
            print(f"[FAIL] {video_id}: could not open video ({e})")
            stats["failed"] += 1
            continue

        sampled_count = 0

        for frame_index_in_video, timestamp, frame in sampler:
            detections = detector.detect(frame)
            if not detections:
                continue

            primary = FaceDetector.select_primary_face(detections, strategy="combined")
            if primary is None:
                continue

            try:
                face_crop = cropper.crop(frame, primary.bbox, margin=0.2)
            except ValueError:
                continue

            filename = save_dir / f"frame_{frame_index_in_video:06d}.png"
            cv2.imwrite(str(filename), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

            rows_by_split[split].append(
                {
                    "video_id": video_id,
                    "frame_index": frame_index_in_video,
                    "timestamp": round(float(timestamp), 6),
                    "face_path": str(filename.relative_to(output_dir.parent if output_dir.parent != output_dir else output_dir)),
                    "label": label,
                }
            )
            sampled_count += 1

        stats["processed"] += 1
        stats["total_faces"] += sampled_count
        print(f"[OK] {video_id}: {sampled_count} faces -> {split} ({'fake' if label == 1 else 'real'})")

    # Write manifests
    for split_name, rows in rows_by_split.items():
        csv_path = output_dir / f"{split_name}.csv"
        fieldnames = ["video_id", "frame_index", "timestamp", "face_path", "label"]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote {len(rows)} rows to {csv_path}")

    # Summary
    print("\n" + "=" * 50)
    print("PREPARATION COMPLETE")
    print("=" * 50)
    print(f"  Videos processed : {stats['processed']}")
    print(f"  Videos skipped   : {stats['skipped']}")
    print(f"  Videos failed    : {stats['failed']}")
    print(f"  Total face crops : {stats['total_faces']}")
    print(f"  Output directory : {output_dir}")
    print("\nUse train.csv, val.csv, test.csv with your dataset loader.")


if __name__ == "__main__":
    main()