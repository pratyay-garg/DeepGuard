import csv
import os
import torchaudio
import torch
import io
from pathlib import Path
from datasets import load_from_disk
import random

def build_split(split_name, asv_reals, asv_fakes):
    print(f"Building {split_name}...")
    in_csv = f"data/processed/celebdf_500/{split_name}.csv"
    out_dir = Path("data/processed/fusion_dataset")
    out_audio_dir = out_dir / "audio"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{split_name}.csv"
    
    # Read rows
    with open(in_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    # Get unique video_ids and their labels
    video_to_label = {}
    for r in rows:
        video_to_label[r["video_id"]] = int(r["label"])
        
    # Assign an audio path to each video_id
    video_to_audio = {}
    for vid, label in video_to_label.items():
        audio_out_path = out_audio_dir / f"{vid}.wav"
        if not audio_out_path.exists():
            # Pick a sample from ASVspoof
            sample = asv_reals.pop() if label == 0 else asv_fakes.pop()
            audio_info = sample["audio"]
            
            try:
                if "bytes" in audio_info:
                    waveform, sr = torchaudio.load(io.BytesIO(audio_info["bytes"]))
                elif "path" in audio_info:
                    waveform, sr = torchaudio.load(audio_info["path"])
                elif "array" in audio_info:
                    waveform = torch.tensor(audio_info["array"], dtype=torch.float32).unsqueeze(0)
                    sr = audio_info["sampling_rate"]
                else:
                    waveform, sr = torch.zeros(1, 16000), 16000
            except Exception as e:
                waveform, sr = torch.zeros(1, 16000), 16000
                
            torchaudio.save(str(audio_out_path), waveform, sr)
        
        video_to_audio[vid] = str(audio_out_path.absolute())
        
    # Write new CSV
    with open(out_csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) + ["audio_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r["audio_path"] = video_to_audio[r["video_id"]]
            writer.writerow(r)
            
    print(f"Finished {split_name}. Videos: {len(video_to_label)}, Rows: {len(rows)}")

def main():
    print("Loading ASVspoof...")
    ds = load_from_disk("data/asvspoof_small")
    reals = [x for x in ds if x["label"] == 0]
    fakes = [x for x in ds if x["label"] == 1]
    
    random.shuffle(reals)
    random.shuffle(fakes)
    
    print(f"ASVspoof Reals: {len(reals)}, Fakes: {len(fakes)}")
    
    build_split("train", reals, fakes)
    build_split("val", reals, fakes)
    build_split("test", reals, fakes)

if __name__ == "__main__":
    main()
