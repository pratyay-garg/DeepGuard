import torch
from src.models.fusion.quick_fusion import QuickFusion

model = QuickFusion()
model.eval()

# batch size 4, sequence length 8, 3 channels, 224x224
video = torch.randn(4, 8, 3, 224, 224)
# batch size 4, 16000 * 4 = 64000 audio samples
audio = torch.randn(4, 64000)

with torch.no_grad():
    out = model(video, audio)
print("Output shape:", out.shape)
