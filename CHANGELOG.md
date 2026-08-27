# Changelog

## [Unreleased]
### Added / Changed (Final Touchups & Reporting)
- **Enhanced Report Generation**: Expanded `scripts/generate_report.py` to extract and visualize additional critical benchmarks (`Accuracy`, `Precision`, and `Recall`) alongside `AUC` and `F1` in both markdown tables and matplotlib charts.
- **Fast Fusion Integration**: Fixed temporal configuration and state dict mappings in `evaluate.py` to successfully evaluate the `fast_fusion` model and seamlessly include it in the final multimodal reports.
- **Audio Evaluation Fixes**: Patched `src/datasets/audio_dataset.py` to gracefully handle corrupted audio samples (preventing data loader crashes) and properly generate `sample_id` metadata to accurately calculate aggregated video-level metrics.

### Added
- **Temporal Localization (Phase 12)**: 
  - Created `scripts/localize.py` that utilizes sliding windows (e.g., 32 frames with 16 frame stride) to localize the precise timing of deepfakes inside videos.
  - Acts as a pure data-router feeding sliced array batches into the existing `evaluate_model` pipeline.
  - Implemented max-score aggregation and suspicious interval detection (outputs timestamps).
  - Updated `MultiFrameDataset` and `MultimodalSequenceDataset` to properly slice unique frames with correct boundary safety checks for sliding windows.
  - Integrated into `commands.sh` via the new `localize` function.

### Phase 11: Audio-Video Synchronization
* Created `AVSyncDataset` inheriting directly from `MultimodalSequenceDataset` to dynamically generate synced (positive) and unsynced (negative) audio-video pairs at runtime without duplicating file loading logic.
* Added `AVSyncModel` implementing a dual-encoder architecture (temporal video backbone and audio backbone) that contrasts features using Cosine Similarity for AV synchronization prediction.
* Added robust model factory routing to cleanly spin up AV Sync using existing backbones (TSM and AudioResNet18).
* Integrated into `train.py` and `evaluate.py` via `is_temporal_multimodal` handling, allowing direct model training and sliding window analysis.

### Phase 13: End-to-End Inference Orchestration
* Built `scripts/infer.py` which takes a single raw `.mp4` video.
* Dynamically extracts audio (via FFmpeg) and crops facial frames (via OpenCV and `FaceCropper`) directly into memory tensors.
* Orchestrates three core models simultaneously (`TSM` for video, `AudioResNet18` for audio, `AVSync` for synchronization).
* Prints an aggregated, user-friendly verdict showing global confidence per modality alongside the sliding-window temporal localization (e.g. `Suspicious interval: 00:06 –------ 00:10`).
* Updated `README.md` to reflect true architecture mappings, datasets (Celeb-DF & DeepfakeTIMIT), and accurate phase completion statuses for world-class readiness.
