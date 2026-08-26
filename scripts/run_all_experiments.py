#!/usr/bin/env python3
import os
import subprocess
import yaml
from pathlib import Path

def run_cmd(cmd):
    print(f"\n[EXEC] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def update_fusion_config(fusion_type, video_model_name):
    """Dynamically updates the yaml config to point to the desired video backbone."""
    config_path = Path(f"configs/{fusion_type}.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["model"]["fusion"]["video_checkpoint"] = f"checkpoints/{video_model_name}/{video_model_name}.pt"
    config["model"]["fusion"]["audio_checkpoint"] = f"checkpoints/audio_resnet18/audio_resnet18.pt"
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    print(f"[CONFIG] Updated {config_path.name} to use {video_model_name}.pt")

def main():
    print("=======================================")
    print(" DEEPGUARD: MASS EXPERIMENT AUTOMATION ")
    print("=======================================")

    # 1. Base Models
    base_models = {
        "single_frame": "train-single",
        "tsm": "train-tsm",
        "temporal_cnn": "train-cnn",
        "bilstm_attention": "train-bilstm",
        "audio_resnet18": "train-audio"
    }

    print("\n--- PHASE 1: Training & Evaluating Base Models ---")
    for exp_name, cmd_suffix in base_models.items():
        run_cmd(f"EXPERIMENT_NAME={exp_name} ./commands.sh {cmd_suffix}")
        run_cmd(f"EXPERIMENT_NAME={exp_name} ./commands.sh eval-{cmd_suffix.split('-')[1]}")

    # 2. Fusion Models
    video_models = ["single_frame", "tsm", "temporal_cnn", "bilstm_attention"]
    fusion_types = ["concat_fusion", "gated_fusion", "cross_attention_fusion"]
    
    # Map the model to a clean suffix for the experiment name (e.g. concat_cnn)
    suffix_map = {
        "single_frame": "single",
        "tsm": "tsm",
        "temporal_cnn": "cnn",
        "bilstm_attention": "bilstm"
    }

    print("\n--- PHASE 2: Training & Evaluating Fusion Combinations ---")
    for f_type in fusion_types:
        f_prefix = f_type.split("_")[0] # 'concat' or 'gated'
        
        for v_model in video_models:
            exp_name = f"{f_prefix}_{suffix_map[v_model]}"
            
            # Update the configuration file dynamically!
            update_fusion_config(f_type, v_model)
            
            # Train and Evaluate
            run_cmd(f"MODEL_TYPE={f_type} EXPERIMENT_NAME={exp_name} ./commands.sh train-fusion")
            run_cmd(f"MODEL_TYPE={f_type} EXPERIMENT_NAME={exp_name} ./commands.sh eval-fusion")

    # 3. Generate Final Report
    print("\n--- PHASE 3: Generating Final Evaluation Report ---")
    run_cmd("./commands.sh generate-report")
    print("\n[SUCCESS] All 13 experiments completed! Check experiments/multimodal_report.md")

if __name__ == "__main__":
    # WARNING: Running this script will take a significant amount of time
    # as it trains 13 distinct models sequentially.
    print("This script will train all possible DeepGuard combinations.")
    resp = input("Are you sure you want to proceed? (y/N): ")
    if resp.lower() == 'y':
        main()
    else:
        print("Aborted.")
