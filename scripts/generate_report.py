import json
import yaml
from pathlib import Path
import math

def generate_report(checkpoints_dir="checkpoints", output_dir="experiments", output_filename="multimodal_report"):
    checkpoints_path = Path(checkpoints_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoints_path.exists():
        print(f"Error: {checkpoints_dir} does not exist.")
        return

    results = []

    for exp_dir in checkpoints_path.iterdir():
        if not exp_dir.is_dir():
            continue

        config_file = exp_dir / "config.yaml"
        metrics_file = exp_dir / f"{exp_dir.name}_frame_metrics.json"

        if not config_file.exists() or not metrics_file.exists():
            continue

        with open(config_file, "r") as f:
            try:
                config = yaml.safe_load(f)
            except Exception:
                continue
                
        with open(metrics_file, "r") as f:
            try:
                metrics = json.load(f)
            except Exception:
                continue

        exp_name = exp_dir.name
        model_name = config.get("model", {}).get("name", "Unknown")
        
        # Determine Fusion Type and Backbones
        if model_name in ["concat_fusion", "gated_fusion", "cross_attention_fusion", "av_sync"]:
            fusion_type = "AVSync" if model_name == "av_sync" else "Concat" if model_name == "concat_fusion" else "Gated" if model_name == "gated_fusion" else "CrossAttention"
            video_ckpt = config.get("model", {}).get("video_checkpoint", "N/A") if model_name == "av_sync" else config.get("model", {}).get("fusion", {}).get("video_checkpoint", "N/A")
            audio_ckpt = config.get("model", {}).get("audio_checkpoint", "N/A") if model_name == "av_sync" else config.get("model", {}).get("fusion", {}).get("audio_checkpoint", "N/A")
            
            # Clean up paths to just basenames for cleaner table
            vid_bb = Path(video_ckpt).name if video_ckpt != "N/A" else "N/A"
            aud_bb = Path(audio_ckpt).name if audio_ckpt != "N/A" else "N/A"
        else:
            fusion_type = "None"
            vid_bb = model_name
            aud_bb = "N/A"

        auc = metrics.get("roc_auc", 0.0)
        f1 = metrics.get("f1", 0.0)
        
        # Keep numeric for plotting
        results.append({
            "exp_name": exp_name,
            "fusion_type": fusion_type,
            "video_bb": vid_bb,
            "audio_bb": aud_bb,
            "auc": auc if not math.isnan(auc) else 0.0,
            "f1": f1 if not math.isnan(f1) else 0.0
        })

    # Sort results to put non-fusion models first, then fusion
    results.sort(key=lambda x: (x["fusion_type"] != "None", x["exp_name"]))

    # Generate Markdown Table
    md_lines = []
    md_lines.append("# DeepGuard Evaluation Report")
    md_lines.append("")
    
    # Embed the visual report image in the markdown
    img_filename = f"{output_filename}.png"
    md_lines.append(f"![Evaluation Metrics]({img_filename})")
    md_lines.append("")
    
    md_lines.append("| Experiment Name | Fusion Type | Video Backbone | Audio Backbone | AUC | F1 |")
    md_lines.append("|-----------------|-------------|----------------|----------------|-----|----|")
    
    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] != 0.0 else "0.0000 / NaN"
        f1_str = f"{r['f1']:.4f}"
        md_lines.append(f"| {r['exp_name']} | {r['fusion_type']} | {r['video_bb']} | {r['audio_bb']} | {auc_str} | {f1_str} |")

    md_output = out_dir / f"{output_filename}.md"
    report_content = "\n".join(md_lines)
    with open(md_output, "w") as f:
        f.write(report_content)
        
    print(report_content)
    print(f"\nMarkdown report saved to {md_output}")

    # Generate Bar Chart with matplotlib
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame(results)
        
        # Set up the plot
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        plt.bar(x - width/2, df["f1"], width, label='F1 Score', color='skyblue')
        plt.bar(x + width/2, df["auc"], width, label='ROC AUC', color='salmon')
        
        plt.xlabel('Experiments', fontweight='bold')
        plt.ylabel('Score (0 to 1)', fontweight='bold')
        plt.title('DeepGuard Multimodal Model Performance Comparison', fontweight='bold', pad=15)
        plt.xticks(x, df["exp_name"], rotation=45, ha="right")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, row in df.iterrows():
            plt.text(i - width/2, row['f1'] + 0.02, f"{row['f1']:.2f}", ha='center', va='bottom', fontsize=9)
            plt.text(i + width/2, row['auc'] + 0.02, f"{row['auc']:.2f}", ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        img_output = out_dir / img_filename
        plt.savefig(img_output, dpi=150)
        print(f"Visual plot saved to {img_output}")
        
    except ImportError:
        print("\nNote: matplotlib or pandas not found. Visual plot was not generated.")
        print("To generate visual plots, run: uv pip install matplotlib pandas")

if __name__ == "__main__":
    generate_report()
