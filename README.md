# DeepGuard

**Efficient multimodal deepfake and manipulated-media detection.**

DeepGuard analyzes video, audio, and audio-visual consistency together to detect manipulated media — and instead of returning a single black-box verdict, it tells you *which* modality looks manipulated and *when* in the timeline it happens.

---

## Why DeepGuard

Most deepfake detectors look at a single modality, usually just the video frames. That leaves a blind spot: a manipulated clip with a convincing face but mismatched or cloned audio can slip past a video-only detector, and vice versa.

DeepGuard scores video and audio independently, then checks whether they agree with each other. The result is a system that can distinguish between three different failure modes instead of collapsing them into one:

- The **video** is manipulated
- The **audio** is manipulated
- The video and audio are individually plausible but **inconsistent** with each other

```
Instead of:              FAKE: 91%

DeepGuard provides:      VIDEO     83%
                         AUDIO     94%
                         AV SYNC   97%
                         Suspicious interval: 
                         00:06 –------ 00:10
```

---

## Architecture

```
                         MEDIA INPUT
                        /            \
                   VIDEO              AUDIO
                     |                  |
            10 FPS Sampling        16 kHz Resample
                     |                  |
               Face Cropping      Log-Mel Spectrogram
                     |                  |
              ResNet18 (frame       AudioResNet18
               embeddings)              |
                     |                  |
          Temporal Model (TSM)          |
                     |                  |
              Video Score          Audio Score
                     \                 /
                       \             /
                    MULTIMODAL FUSION
                 (Concat / Gated / AV Sync)
                            |
                   (Cross-Attention)
                            |
                    REAL / MANIPULATED
                  Confidence + Timestamped
                 Manipulation Windows (~3.2s)
```

**Design principle:** Audio and video are processed into temporal embeddings independently. They meet at various fusion layers (Concat, Gated, Cross-Attention) and an AV-Sync contrastive module, which maintains flexibility and robust deepfake detection capabilities.

---

## Pipeline Breakdown

| Stage | What it does |
|---|---|
| **Spatial** | Extract face regions, sample at 10 FPS to cut redundant computation, learn facial textures with ResNet18. |
| **Temporal** | Temporal Shift Module (TSM) analyzes short frame sequences (overlapping 3.2s windows). |
| **Audio** | 16 kHz audio → log-mel spectrogram → AudioResNet18 to catch synthetic speech artifacts. |
| **Fusion** | Independent video + audio scores merged via flexible multimodal fusion pipelines. |
| **AV Sync** | Contrastive AV Sync module calculates Cosine Similarity between Audio/Video embeddings. |
| **Localization** | Sliding 3.2-second windows → timestamped manipulation probability (localize the deepfake). |

---

## Tech Stack

- **Language / Framework:** Python, PyTorch
- **Vision:** OpenCV, TorchVision
- **Audio:** Librosa / TorchAudio
- **ML utilities:** Scikit-learn
- **Data:** Celeb-DF-v2, DeepfakeTIMIT (Preprocessed manifests)

---

## Implementation Status

| Phase | Status |
|---|---|
| 1-2 | ✅ Complete — Single-frame ResNet18 & Temporal modeling (TSM) |
| 3 | ✅ Complete — Audio Baseline (AudioResNet18 & Mel-Spectrograms) |
| 4-10 | ✅ Complete — Multimodal Fusion (Concat, Gated, Cross-Attention) |
| 11 | ✅ Complete — Audio-Video Synchronization (Contrastive Learning) |
| 12 | ✅ Complete — Temporal Localization (Sliding Windows ~3.2s) |

### Current Features

- **End-to-End Orchestration (`infer.py`)** — Unified inference on raw `.mp4` files.
- **Multimodal Evaluation** — Analyzes Video, Audio, and AV Sync simultaneously.
- **Temporal Localization** — Outputs exact suspicious time intervals in the media.
- **Advanced Fusion Architectures** — Concat, Gated, Cross-Attention, and AV Sync.

---

## Datasets

- [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) — Visual manipulation baseline (used processed subset)
- [DeepfakeTIMIT](https://www.idiap.ch/dataset/deepfaketimit) — Audio-visual detection & multimodal analysis


## Feasibility & Known Challenges

| Challenge | Mitigation |
|---|---|
| High computational cost | Efficient frame sampling + face-level processing |
| Different audio/video time scales | Common timestamp-based segmentation |
| Compression & low-quality media | Augmentation across multiple compression levels |
| New/unknown manipulation methods | Cross-dataset evaluation + multimodal cues |
| False positives | Multi-modal evidence + confidence calibration |
| Audio-video misalignment | Explicit temporal synchronization module |
| Dataset bias | Train/test separation at video/identity level |

---

## Target Use Cases

- Social-media / content platforms (moderation)
- News & fact-checking organizations
- Cybersecurity teams
- Digital-forensics investigators
- Individuals verifying suspicious media

Deployable as a **web/API service**, a **content moderation pipeline**, a **digital-forensics tool**, or a **batch media analysis system**.

---

## References

- Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images* — [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- He et al., *Deep Residual Learning for Image Recognition (ResNet)* — [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- Lin et al., *TSM: Temporal Shift Module for Efficient Video Understanding* — [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html)
- SCRFD — *Sample and Computation Redistribution for Efficient Face Detection* — [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- DFDC — Facebook/Meta Deepfake Detection Challenge Dataset — [ai.meta.com/datasets/dfdc](https://ai.meta.com/datasets/dfdc/)
- ASVspoof 2021 — [asvspoof.org](https://www.asvspoof.org/index2021.html)
- AV-Deepfake1M — [openreview.net](https://openreview.net/forum?id=YZ68Ifi4yH)

---

