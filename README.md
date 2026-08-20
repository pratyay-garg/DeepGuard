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
         Lightweight Face Detection   Log-Mel Spectrogram
              (SCRFD, 224x224)             |
                     |                Lightweight CNN
              ResNet18 (frame            |
               embeddings)          Audio Score
                     |                   |
          Temporal Model (TSM) 
                     |                   |
              Video Score
                     \                 /
                        GATED FUSION
                            |
               (Cross-Attention — optional,
               triggered only if it improves
               accuracy enough to justify cost)
                            |
                Audio-Visual Consistency
                            |
                    REAL / MANIPULATED
                  Confidence + Timestamped
                 Manipulation Windows (~3.2s)
```

**Design principle:** audio and video are scored independently first (cheap, fast), and only meet at a gated fusion layer. The more expensive cross-attention fusion is a later, optional stage — not a hard dependency — which keeps the pipeline usable on modest compute.

---

## Pipeline Breakdown

| Stage | What it does |
|---|---|
| **Spatial** | Extract face regions with SCRFD, sample at 10 FPS to cut redundant computation, learn facial textures with ResNet18 |
| **Temporal** | Temporal Shift Module (TSM) analyzes short frame sequences — similar temporal awareness to an LSTM, far less overhead |
| **Audio** | 16 kHz audio → log-mel spectrogram → lightweight CNN to catch synthetic speech / voice-cloning artifacts |
| **Fusion** | Independent video + audio scores merged via gated fusion; cross-attention fusion triggered only when justified |
| **Localization** | 3.2-second windows → timestamped manipulation probability, not just a single verdict |

---

## Tech Stack

- **Language / Framework:** Python, PyTorch
- **Vision:** OpenCV, TorchVision
- **Audio:** Librosa / TorchAudio
- **ML utilities:** Scikit-learn
- **Face detection:** SCRFD (lightweight)

---

## Implementation Roadmap

| Phase | Goal |
|---|---|
| 1 | Visual baseline — Face crop → ResNet18 → Real/Fake |
| 2 | Temporal modelling — Frame embeddings → Temporal CNN / TSM |
| 3 | Audio baseline — Audio → Mel spectrogram → CNN → Real/Fake |
| 4 | Multimodal fusion — Video + Audio → Gated Fusion |
| 5 | Advanced fusion — Video + Audio → Cross-Attention |
| 6 | Localization — 3.2s windows → timestamped manipulation probability |

This repo currently reflects the **idea/architecture submission stage** for OMNIKON 2025; phases above track planned implementation order.

---

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — visual manipulation baseline
- [DFDC](https://ai.meta.com/datasets/dfdc/) — generalization / cross-dataset evaluation
- [ASVspoof 2021](https://www.asvspoof.org/index2021.html) — audio spoofing benchmark
- [AV-Deepfake1M](https://openreview.net/forum?id=YZ68Ifi4yH) — audio-visual detection & temporal localization

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

