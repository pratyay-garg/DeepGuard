import argparse
import json
import math
import subprocess
import tempfile
import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import src even if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import torch.nn.functional as F

from src.models.model_factory import create_model
from src.preprocessing.face_detector import OpenCVFaceDetector, FaceDetector
from src.preprocessing.face_cropper import FaceCropper
from src.datasets.transforms import build_eval_transforms
from src.preprocessing.audio import compute_mel_spectrogram
from src.utils.device import get_device
import torchaudio
import numpy as np


def extract_audio(video_path, output_path):
    subprocess.run([
        'ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '16000', '-ac', '1', str(output_path), '-y'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_frames(video_path, target_fps=10.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25.0
    
    frame_interval = max(1, int(fps / target_fps))
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        frame_idx += 1
        
    cap.release()
    return frames

def load_video_model(ckpt_path, device):
    config = {
        "name": "resnet18",
        "pretrained": False,
        "num_classes": 2,
        "temporal": {
            "enabled": True,
            "model": "tsm",
            "sequence_length": 32
        }
    }
    model = create_model(config).to(device)
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    return model

def load_audio_model(ckpt_path, device):
    config = {
        "name": "audio_resnet18",
        "pretrained": False,
        "num_classes": 2,
        "embedding_dim": 512
    }
    model = create_model(config).to(device)
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    return model

def load_sync_model(ckpt_path, device):
    config = {
        "name": "av_sync",
        "video_embedding_dim": 512,
        "audio_embedding_dim": 512,
        "projection_dim": 256,
        "sequence_length": 16
    }
    model = create_model(config).to(device)
    if Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="End-to-End DeepGuard Inference")
    parser.add_argument("video_path", type=str, help="Path to input .mp4 file")
    parser.add_argument("--video-ckpt", type=str, default="checkpoints/tsm/tsm.pt", help="Path to video model")
    parser.add_argument("--audio-ckpt", type=str, default="checkpoints/audio_resnet18/audio_resnet18.pt", help="Path to audio model")
    parser.add_argument("--sync-ckpt", type=str, default="checkpoints/av_sync_1/av_sync_1.pt", help="Path to sync/fusion model")
    parser.add_argument("--sync-model-name", type=str, default="av_sync", help="Name of sync/fusion model architecture")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return

    device = get_device()
    
    # Load Models
    video_model = load_video_model(args.video_ckpt, device)
    audio_model = load_audio_model(args.audio_ckpt, device)
    
    config = {
        "name": args.sync_model_name,
        "video_embedding_dim": 512,
        "audio_embedding_dim": 512,
        "projection_dim": 256,
        "sequence_length": 16
    }
    sync_model = create_model(config).to(device)
    if Path(args.sync_ckpt).exists():
        state = torch.load(args.sync_ckpt, map_location=device)
        state = state.get("model_state_dict", state)
        # Map keys if fast_fusion was trained separately
        if args.sync_model_name == "fast_fusion" and not any(k.startswith("fusion_head.") for k in state.keys()):
            state = {f"fusion_head.{k}": v for k, v in state.items()}
        sync_model.load_state_dict(state, strict=False)
    sync_model.eval()

    # Face detection
    face_detector = OpenCVFaceDetector(conf_threshold=0.6)
    face_cropper = FaceCropper(target_size=(224, 224))
    transform = build_eval_transforms(224)
    
    print("Preprocessing video...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audio_path = tmpdir_path / "audio.wav"
        extract_audio(video_path, audio_path)
        
        frames = extract_frames(video_path)
        
        if not frames:
            print("Failed to extract frames.")
            return

        face_tensors = []
        for frame in frames:
            detections = face_detector.detect(frame)
            if detections:
                primary = FaceDetector.select_primary_face(detections, strategy="combined")
                face_crop = face_cropper.crop(frame, primary.bbox, margin=0.2)
            else:
                h, w = frame.shape[:2]
                sz = min(h, w)
                face_crop = frame[(h-sz)//2:(h+sz)//2, (w-sz)//2:(w+sz)//2]
                face_crop = cv2.resize(face_crop, (224, 224))
            
            # Convert to PIL Image for torchvision transforms
            from PIL import Image
            img = Image.fromarray(face_crop)
            tensor = transform(img)
            face_tensors.append(tensor)
            
        if not face_tensors:
            print("No valid faces found.")
            return
            
        face_seq = torch.stack(face_tensors) # [Total_frames, C, H, W]

        # Extract Audio Spectrogram
        if audio_path.exists():
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            spectrogram = compute_mel_spectrogram(
                waveform, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=2.0
            )
        else:
            spectrogram = torch.zeros((1, 80, 200))

    # We need to run sliding windows.
    # Video Model: 32 frames
    vid_seq_len = 32
    # Sync Model: 16 frames
    sync_seq_len = 16
    
    fps = 10.0
    
    video_scores = []
    audio_scores = []
    sync_scores = []
    
    intervals = []
    
    with torch.no_grad():
        total_frames = face_seq.shape[0]
        stride = 16
        
        # Audio global score
        audio_input = spectrogram.unsqueeze(0).to(device) # [1, 1, 80, T]
        # truncate or pad audio to 200 for model if needed, but AudioResNet18 has adaptive pool
        a_logits = audio_model.predict(audio_input)
        a_prob = F.softmax(a_logits, dim=1)[0, 1].item()
        
        for i in range(0, max(1, total_frames - vid_seq_len + 1), stride):
            end_idx = min(i + vid_seq_len, total_frames)
            window = face_seq[i:end_idx]
            
            # Pad if needed
            if window.shape[0] < vid_seq_len:
                pad_size = vid_seq_len - window.shape[0]
                window = torch.cat([window, window[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
                
            v_input = window.unsqueeze(0).to(device) # [1, T, C, H, W]
            
            v_logits = video_model(v_input)
            v_prob = F.softmax(v_logits, dim=1)[0, 1].item()
            video_scores.append(v_prob)
            
            start_time = i / fps
            end_time = end_idx / fps
            intervals.append({"start": start_time, "end": end_time, "v_prob": v_prob})
            
        for i in range(0, max(1, total_frames - sync_seq_len + 1), stride):
            end_idx = min(i + sync_seq_len, total_frames)
            window = face_seq[i:end_idx]
            
            if window.shape[0] < sync_seq_len:
                pad_size = sync_seq_len - window.shape[0]
                window = torch.cat([window, window[-1:].repeat(pad_size, 1, 1, 1)], dim=0)
                
            v_input = window.unsqueeze(0).to(device)
            # Find matching audio segment
            # 16 frames at 10fps = 1.6s. Audio hop is 10ms (100 Hz).
            # 1.6s = 160 frames in spectrogram
            start_time = i / fps
            end_time = end_idx / fps
            
            start_hop = int(start_time * 100)
            end_hop = int(end_time * 100)
            
            if end_hop > spectrogram.shape[2]:
                end_hop = spectrogram.shape[2]
                
            s_audio = spectrogram[:, :, start_hop:end_hop]
            if s_audio.shape[2] < 160:
                pad_amt = 160 - s_audio.shape[2]
                s_audio = F.pad(s_audio, (0, pad_amt))
                
            s_audio = s_audio.unsqueeze(0).to(device)
            
            sync_logits = sync_model(v_input, s_audio)
            # Class 0: unsync (fake), Class 1: sync (real)
            # So fake probability is class 0!
            sync_prob = F.softmax(sync_logits, dim=1)[0, 0].item() 
            sync_scores.append(sync_prob)
            
            # Match to interval
            for it in intervals:
                if it["start"] <= start_time and it["end"] >= end_time:
                    if "s_prob" not in it or sync_prob > it["s_prob"]:
                        it["s_prob"] = sync_prob

    final_v = max(video_scores) if video_scores else 0.0
    final_a = a_prob
    final_s = max(sync_scores) if sync_scores else 0.0
    
    print("\n" + "="*50)
    print(f"DeepGuard provides:      VIDEO     {int(final_v * 100)}%")
    print(f"                         AUDIO     {int(final_a * 100)}%")
    print(f"                         AV SYNC   {int(final_s * 100)}%")
    
    # Find most suspicious interval combining scores
    max_score = 0
    sus_interval = None
    for it in intervals:
        score = max(it.get("v_prob", 0), it.get("s_prob", 0), final_a)
        if score > max_score:
            max_score = score
            sus_interval = it
            
    if sus_interval and max_score > 0.5:
        start_s = int(sus_interval['start'])
        end_s = int(sus_interval['end'])
        print(f"                         Suspicious interval:")
        print(f"                         {start_s//60:02d}:{start_s%60:02d} –------ {end_s//60:02d}:{end_s%60:02d}")
    else:
        print(f"                         Suspicious interval:")
        print(f"                         None detected")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
