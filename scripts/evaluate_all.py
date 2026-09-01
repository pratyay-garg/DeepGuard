import os
import glob
import json
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate all epoch checkpoints in a folder and find the best one.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing the .pt files")
    parser.add_argument("--config", type=str, default="configs/fast_fusion.yaml", help="Path to config file")
    parser.add_argument("--manifest", type=str, default="data/processed/fakeavceleb/val.csv", help="Path to val.csv")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Directory {checkpoint_dir} does not exist.")
        return

    # Find all fast_fusion_epoch_X.pt files
    checkpoints = sorted(glob.glob(str(checkpoint_dir / "*_epoch_*.pt")))
    
    if not checkpoints:
        print(f"No epoch checkpoints found in {checkpoint_dir}.")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate.\\n")
    
    best_f1 = -1
    best_checkpoint = None
    results = {}

    for ckpt in checkpoints:
        print(f"Evaluating {os.path.basename(ckpt)}...")
        
        # Run the evaluation script
        cmd = [
            "uv", "run", "python", "-m", "scripts.evaluate",
            "--config", args.config,
            "--manifest", args.manifest,
            "--checkpoint", ckpt
        ]
        
        # In case 'uv' is not installed, fallback to standard python
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            cmd = cmd[2:] # Remove 'uv', 'run'
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {ckpt}:\\n{e.stderr.decode('utf-8')}")
            continue

        # Parse the JSON results
        ckpt_path = Path(ckpt)
        metrics_file = ckpt_path.parent / f"{ckpt_path.stem}_video_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                f1_score = metrics.get('f1', 0)
                auc_score = metrics.get('roc_auc', 0)
                precision = metrics.get('precision', 0)
                accuracy = metrics.get('accuracy', 0)
                
                results[os.path.basename(ckpt)] = f1_score
                print(f" -> F1: {f1_score:.4f} | AUC: {auc_score:.4f} | Precision: {precision:.4f} | Accuracy: {accuracy:.4f}")
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_checkpoint = os.path.basename(ckpt)
        else:
            print(f" -> Failed to find metrics file for {ckpt}")

    print("\\n" + "="*50)
    if best_checkpoint:
        print(f"🏆 BEST MODEL: {best_checkpoint}")
        print(f"🏆 BEST F1 SCORE: {best_f1:.4f}")
    else:
        print("No valid results found.")
    print("="*50)

if __name__ == "__main__":
    main()
