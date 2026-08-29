import csv
import random
from pathlib import Path
from datasets import load_from_disk
import os

def main():
    print("Building unified multimodal dataset...")
    celeb_csv = "data/processed/celebdf_500/train.csv"
    
    # 1. Load CelebDF frames
    real_video_frames = []
    fake_video_frames = []
    with open(celeb_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "0":
                real_video_frames.append(row["face_path"])
            else:
                fake_video_frames.append(row["face_path"])
    
    print(f"Loaded {len(real_video_frames)} real video frames, {len(fake_video_frames)} fake video frames.")
    
    # Group frames by video_id to preserve temporal sequences if needed, but for now we just want paths.
    # Actually, AVSyncDataset and MultimodalSequenceDataset group by video_id internally!
    # Let's see how MultimodalSequenceDataset parses. It reads 'video_id', 'frame_index', 'face_path', 'audio_path', 'label'
