import os
import sys

# Change to model_submit directory so relative paths work (like checkpoints/)
os.chdir('model_submit')
sys.path.append(os.getcwd())

from app import predict

video_path = "../data/raw/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/Asian (South)/men/id00032/00028_0_id00860_wavtolip.mp4"
result = predict(video_path)
print("Prediction Result:")
print(result)
