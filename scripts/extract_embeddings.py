import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.multimodal_dataset import MultimodalSequenceDataset
from src.datasets.transforms import build_eval_transforms
from src.models.audio.audio_resnet18 import AudioResNet18
from src.utils.device import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/deepfaketimit_multimodal/train.csv")
    parser.add_argument("--out-dir", type=str, default="data/embeddings")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load frozen standard backbones
    print("Loading ResNet18 and AudioResNet18 (Frozen)...")
    v_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    v_backbone.fc = torch.nn.Identity() # output 512-d
    v_backbone = v_backbone.to(device).eval()

    a_backbone = AudioResNet18(pretrained=True, num_classes=2)
    # the extract_embedding method bypasses the classifier
    a_backbone = a_backbone.to(device).eval()

    # Load Dataset
    transform = build_eval_transforms(224)
    dataset = MultimodalSequenceDataset(
        args.manifest,
        transform=transform,
        audio_transform=None,
        sequence_length=args.sequence_length,
        stride=args.stride
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    
    out_dir = Path(args.out_dir)
    split_name = Path(args.manifest).stem # 'train', 'val', or 'test'
    v_dir = out_dir / split_name / "video"
    a_dir = out_dir / split_name / "audio"
    v_dir.mkdir(parents=True, exist_ok=True)
    a_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    
    print(f"Extracting embeddings for {len(dataset)} sequences...")
    
    with torch.no_grad():
        global_idx = 0
        for batch_idx, (v_seq, a_seq, label, metadata) in enumerate(tqdm(loader)):
            # v_seq: [B, T, C, H, W]
            # a_seq: [B, 1, F, T_a]
            B, T_v, C, H, W = v_seq.shape
            
            # Extract Video Embeddings (frame by frame)
            v_seq_flat = v_seq.view(B * T_v, C, H, W).to(device)
            v_emb_flat = v_backbone(v_seq_flat) # [B*T, 512]
            v_emb = v_emb_flat.view(B, T_v, 512)
            
            # Extract Audio Embeddings (global for the sequence window)
            a_seq = a_seq.to(device)
            a_emb = a_backbone.extract_embedding(a_seq) # [B, 512]
            
            v_emb = v_emb.cpu()
            a_emb = a_emb.cpu()
            
            for i in range(B):
                torch.save(v_emb[i], v_dir / f"{global_idx}.pt")
                torch.save(a_emb[i], a_dir / f"{global_idx}.pt")
                labels.append({"id": global_idx, "label": label[i].item()})
                global_idx += 1
                
    df = pd.DataFrame(labels)
    df.to_csv(out_dir / split_name / "labels.csv", index=False)
    print(f"Saved {global_idx} embeddings to {out_dir}/{split_name}")

if __name__ == "__main__":
    main()
