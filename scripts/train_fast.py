import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.datasets.embedding_dataset import EmbeddingDataset
from src.models.fusion.fast_fusion import FastCrossAttentionFusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/embeddings/train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Load Fast Dataset
    dataset = EmbeddingDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Init Fast CPU Model
    model = FastCrossAttentionFusion()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting extremely fast CPU training on {len(dataset)} embeddings...")
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for v_emb, a_emb, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            logits = model(v_emb, a_emb)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Acc: {correct/total:.4f}")
        
    print("Training complete! Your laptop CPU survived!")
    Path("checkpoints/fast_fusion").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fast_fusion/fast_fusion.pt")
    
if __name__ == "__main__":
    main()
