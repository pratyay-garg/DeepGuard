import csv
from pathlib import Path

def rebuild(split_name):
    base_dir = Path(f"data/processed/celebdf_500")
    split_dir = base_dir / "faces" / split_name
    out_csv = base_dir / f"{split_name}.csv"
    
    rows = []
    
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        label_dir = split_dir / label_name
        if not label_dir.exists(): continue
        
        for video_dir in label_dir.iterdir():
            if not video_dir.is_dir(): continue
            video_id = video_dir.name
            
            for face_file in video_dir.glob("frame_*.png"):
                # face_file.name is like frame_000005.png
                frame_idx = int(face_file.stem.split("_")[1])
                # dummy timestamp, we can just do frame_idx / 30.0
                timestamp = frame_idx / 30.0
                
                rel_path = face_file.relative_to(base_dir.parent)
                
                rows.append({
                    "video_id": video_id,
                    "frame_index": frame_idx,
                    "timestamp": timestamp,
                    "face_path": str(rel_path),
                    "label": label_val
                })
                
    # Sort by video_id, then frame_index
    rows.sort(key=lambda x: (x["video_id"], x["frame_index"]))
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "frame_index", "timestamp", "face_path", "label"])
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Rebuilt {out_csv} with {len(rows)} rows.")

rebuild("train")
rebuild("val")
