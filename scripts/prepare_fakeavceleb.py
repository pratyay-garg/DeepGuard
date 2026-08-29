import argparse
import csv
import hashlib
import random
import subprocess
from pathlib import Path

import cv2

from src.preprocessing.video_sampler import VideoSampler
from src.preprocessing.face_detector import OpenCVFaceDetector, FaceDetector
from src.preprocessing.face_cropper import FaceCropper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="data/raw/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2")
    parser.add_argument("--output-dir", type=str, default="data/processed/fakeavceleb")
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--max-videos-per-class", type=int, default=500)
    return parser.parse_args()

def extract_audio(video_path, output_path):
    subprocess.run([
        'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '16000', '-ac', '1', str(output_path), '-y'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def split_name_for_video(video_id: str) -> str:
    digest = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    if value < 0.8: return "train"
    elif value < 0.9: return "val"
    else: return "test"

def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    real_vids = list((dataset_root / "RealVideo-RealAudio").rglob("*.mp4"))
    fake_vids = []
    fake_vids.extend(list((dataset_root / "FakeVideo-FakeAudio").rglob("*.mp4")))
    fake_vids.extend(list((dataset_root / "FakeVideo-RealAudio").rglob("*.mp4")))
    fake_vids.extend(list((dataset_root / "RealVideo-FakeAudio").rglob("*.mp4")))
    
    random.seed(42)
    random.shuffle(real_vids)
    random.shuffle(fake_vids)
    
    if args.max_videos_per_class:
        real_vids = real_vids[:args.max_videos_per_class]
        fake_vids = fake_vids[:args.max_videos_per_class]
        
    all_videos = [(v, 0) for v in real_vids] + [(v, 1) for v in fake_vids]
    
    print(f"Processing {len(real_vids)} reals and {len(fake_vids)} fakes...")
    
    detector = OpenCVFaceDetector(conf_threshold=0.6)
    cropper = FaceCropper(target_size=(224, 224))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    
    rows_by_split = {"train": [], "val": [], "test": []}
    
    for video_path, label in all_videos:
        video_id = video_path.stem + "_" + video_path.parent.name
        split = split_name_for_video(video_id)
        
        save_dir = output_dir / "faces" / split / ("fake" if label == 1 else "real") / video_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = audio_out_dir / f"{video_id}.wav"
        if not audio_path.exists():
            extract_audio(video_path, audio_path)
            
        try:
            sampler = VideoSampler(str(video_path), target_fps=args.target_fps)
        except Exception as e:
            print(f"[FAIL] {video_id}: could not open video ({e})")
            continue
            
        sampled_count = 0
        for frame_index, timestamp, frame in sampler:
            detections = detector.detect(frame)
            if not detections: continue
            
            primary = FaceDetector.select_primary_face(detections, strategy="combined")
            if primary is None: continue
            
            try:
                face_crop = cropper.crop(frame, primary.bbox, margin=0.2)
            except ValueError:
                continue
                
            filename = save_dir / f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(filename), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            
            rel_face_path = str(filename.relative_to(output_dir))
            
            rows_by_split[split].append({
                "video_id": video_id,
                "frame_index": frame_index,
                "timestamp": round(float(timestamp), 6),
                "face_path": rel_face_path,
                "label": label,
                "audio_path": str(audio_path.absolute())
            })
            sampled_count += 1
            
        print(f"[OK] {video_id}: {sampled_count} frames extracted -> {split}")
        
    for split_name, rows in rows_by_split.items():
        csv_path = output_dir / f"{split_name}.csv"
        fieldnames = ["video_id", "frame_index", "timestamp", "face_path", "label", "audio_path"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"Wrote {len(rows)} rows to {csv_path}")

if __name__ == "__main__":
    main()
