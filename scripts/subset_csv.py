import csv
import random
from pathlib import Path

def subset(in_csv, out_csv, n_videos_per_class=20):
    with open(in_csv, "r") as f:
        rows = list(csv.DictReader(f))
        
    vids_by_class = {0: set(), 1: set()}
    for r in rows:
        vids_by_class[int(r["label"])].add(r["video_id"])
        
    keep_vids = set()
    for lbl in [0, 1]:
        vids = list(vids_by_class[lbl])
        random.shuffle(vids)
        keep_vids.update(vids[:n_videos_per_class])
        
    keep_rows = [r for r in rows if r["video_id"] in keep_vids]
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(keep_rows)
        
    print(f"Subset {in_csv} -> kept {len(keep_vids)} videos, {len(keep_rows)} rows.")

subset("data/processed/celebdf_500/train.csv", "data/processed/celebdf_500/train.csv", n_videos_per_class=20)
subset("data/processed/celebdf_500/val.csv", "data/processed/celebdf_500/val.csv", n_videos_per_class=5)
subset("data/processed/fusion_dataset/train.csv", "data/processed/fusion_dataset/train.csv", n_videos_per_class=20)
subset("data/processed/fusion_dataset/val.csv", "data/processed/fusion_dataset/val.csv", n_videos_per_class=5)
