from __future__ import annotations

import argparse
import csv
import hashlib
import random
import subprocess
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DeepfakeTIMIT dataset for DeepGuard with audio.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/raw/DeepfakeTIMIT",
        help="Root directory of DeepfakeTIMIT videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/deepfaketimit_multimodal",
        help="Where to write crops, manifests, and audio.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=10,
        help="Sampling rate (frames per second).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Max number of videos to process. None = use all.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
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

    random.seed(42)  # Fixed seed for splitting
    digest = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)

    if value < train_ratio:
        return "train"
    elif value < train_ratio + val_ratio:
        return "val"
    else:
        return "test"


def discover_deepfaketimit_videos(dataset_root: Path, max_videos: int | None, seed: int):
    if not dataset_root.exists():
        raise FileNotFoundError(f"DeepfakeTIMIT dataset root not found: {dataset_root}")

    videos = []
    quality_folders = ["higher_quality", "lower_quality"]

    for folder_name in quality_folders:
        folder_path = dataset_root / folder_name
        if folder_path.exists():
            for video_path in sorted(folder_path.rglob("*.avi")):
                videos.append(video_path)

    print(f"Found {len(videos)} DeepfakeTIMIT videos.")

    # Optional subsampling
    random.seed(seed)
    if max_videos is not None and max_videos < len(videos):
        videos = random.sample(videos, max_videos)
        print(f"  -> Sampled {max_videos} videos (seed={seed})")

    if not videos:
        raise FileNotFoundError(
            f"No DeepfakeTIMIT videos found under {dataset_root}. "
            "Expected folders: higher_quality, lower_quality"
        )

    return videos


def is_fake_video(video_path: Path) -> bool:
    """
    DeepfakeTIMIT: All videos are fake (face-swapped).
    Videos in higher_quality/ and lower_quality/ are both fake.
    """
    return True  # All DeepfakeTIMIT videos are fakes


def build_video_id(video_path: Path, dataset_root: Path) -> str:
    rel = video_path.relative_to(dataset_root)
    return rel.with_suffix("").as_posix().replace("/", "__")


def extract_audio(video_path: Path, output_dir: Path) -> Path | None:
    """
    Extract audio from video and save as WAV file.
    Returns path to audio file or None if extraction fails.
    """
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"{Path(video_path).stem}.wav"

    if audio_path.exists():
        return audio_path

    print(f"  Extracting audio: {video_path.name} -> {audio_path.name}")
    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",  # Overwrite if exists
            str(audio_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f"  Warning: Failed to extract audio: {result.stderr}")
            return None

        # Verify audio file was created
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return None

        return audio_path

    except Exception as e:
        print(f"  Warning: Could not extract audio: {e}")
        return None


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    from src.preprocessing.face_cropper import FaceCropper
    from src.preprocessing.face_detector import OpenCVFaceDetector, FaceDetector

    detector = OpenCVFaceDetector(conf_threshold=0.6)
    cropper = FaceCropper(target_size=(224, 224))

    videos = discover_deepfaketimit_videos(
        dataset_root,
        max_videos=args.max_videos,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {"train": [], "val": [], "test": []}
    stats = {"processed": 0, "skipped": 0, "failed": 0, "total_faces": 0, "total_audio": 0}

    for video_path in videos:
        label = 1 if is_fake_video(video_path) else 0
        video_id = build_video_id(video_path, dataset_root)
        split = split_name_for_video(video_id, args.train_ratio, args.val_ratio, args.test_ratio)

        save_dir = output_dir / "faces" / split / "fake" / video_id

        # Skip if already processed and --skip-existing is set
        if args.skip_existing and save_dir.exists() and any(save_dir.iterdir()):
            print(f"[SKIP] {video_id}: already processed")
            stats["skipped"] += 1
            continue

        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            from src.preprocessing.video_sampler import VideoSampler
            sampler = VideoSampler(str(video_path), target_fps=args.target_fps)
        except Exception as e:
            print(f"[FAIL] {video_id}: could not open video ({e})")
            stats["failed"] += 1
            continue

        # Extract audio
        audio_path = extract_audio(video_path, output_dir)
        if audio_path:
            stats["total_audio"] += 1

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
                    "audio_path": str(audio_path) if audio_path else "",
                }
            )
            sampled_count += 1
            stats["total_faces"] += 1

        stats["processed"] += 1

    # Write manifests
    for split_name, rows in rows_by_split.items():
        csv_path = output_dir / f"{split_name}.csv"
        fieldnames = ["video_id", "frame_index", "timestamp", "face_path", "label", "audio_path"]

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
    print(f"  Audio files      : {stats['total_audio']}")
    print(f"  Output directory : {output_dir}")
    print("\nUse train.csv, val.csv, test.csv with your multimodal dataset loader.")


if __name__ == "__main__":
    main()