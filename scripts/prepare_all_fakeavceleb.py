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
    parser.add_argument("--output-dir", type=str, default="data/processed/fakeavceleb_all")
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--max-videos-per-class", type=int, default=None, help="Set to limit videos per class. Leave None for all.")
    return parser.parse_args()

def extract_audio(video_path, output_path):
    subprocess.run([
        'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '16000', '-ac', '1', str(output_path), '-y'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def split_name_for_video(video_id: str) -> str:
    digest = hashlib.md5(video_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    if value < 0.7:
        return "train"
    elif value < 0.85:
        return "val"
    else:
        return "test"

def main():
    args = parse_args()
    root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    real_dir = root / "RealVideo-RealAudio"
    fake_dirs = [root / "FakeVideo-FakeAudio", root / "FakeVideo-RealAudio"]
    
    real_vids = list(real_dir.rglob("*.mp4"))
    fake_vids = []
    for fd in fake_dirs:
        fake_vids.extend(list(fd.rglob("*.mp4")))
        
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
    
    manifest_rows = []
    
    for vpath, label in all_videos:
        split = split_name_for_video(vpath.stem)
        
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            continue
            
        frame_stride = max(1, int(fps / args.target_fps))
        
        frame_idx = 0
        saved_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_stride == 0:
                faces = detector.detect(frame)
                if faces:
                    best_face = max(faces, key=lambda f: f.confidence)
                    cropped = cropper.crop(frame, best_face.bbox)
                    
                    vid_out_dir = output_dir / split / vpath.stem
                    vid_out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = vid_out_dir / f"frame_{frame_idx:04d}.jpg"
                    
                    cv2.imwrite(str(out_path), cropped)
                    saved_frames.append(str(out_path.relative_to(output_dir)))
            frame_idx += 1
            
        cap.release()
        
        if saved_frames:
            audio_out_path = audio_out_dir / f"{vpath.stem}.wav"
            extract_audio(vpath, audio_out_path)
            
            # Use relative paths for portability
            manifest_rows.append({
                "video_id": vpath.stem,
                "label": label,
                "split": split,
                "frame_paths": "|".join(saved_frames),
                "audio_path": str(audio_out_path.relative_to(output_dir))
            })
            
    print(f"Successfully processed {len(manifest_rows)} videos")
    
    # Write manifests
    for split in ["train", "val", "test"]:
        split_rows = [r for r in manifest_rows if r["split"] == split]
        with open(output_dir / f"{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "label", "split", "frame_paths", "audio_path"])
            writer.writeheader()
            writer.writerows(split_rows)
            
    print("All manifests written.")

if __name__ == "__main__":
    main()
