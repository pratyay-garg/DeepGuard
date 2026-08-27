import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """
    Dataset that loads pre-computed 1D embedding vectors instead of raw images/audio.
    This bypasses ResNet bottlenecks, making CPU training lighting fast.
    """
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.labels_df = pd.read_csv(self.data_dir / "labels.csv")
        self.v_dir = self.data_dir / "video"
        self.a_dir = self.data_dir / "audio"
        
    def __len__(self):
        return len(self.labels_df)
        
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        emb_id = row['id']
        label = row['label']
        
        v_emb = torch.load(self.v_dir / f"{emb_id}.pt", weights_only=True) # [T, 512]
        a_emb = torch.load(self.a_dir / f"{emb_id}.pt", weights_only=True) # [512]
        
        return v_emb, a_emb, label
