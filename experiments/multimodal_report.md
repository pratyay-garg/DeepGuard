# DeepGuard Evaluation Report

![Evaluation Metrics](multimodal_report.png)

| Experiment Name | Fusion Type | Video Backbone | Audio Backbone | Accuracy | Precision | Recall | F1 | AUC |
|-----------------|-------------|----------------|----------------|----------|-----------|--------|----|-----|
| audio_resnet18 | None | audio_resnet18 | N/A | 0.9205 | 0.9927 | 0.9242 | 0.9572 | 0.9212 |
| tsm | None | resnet18 | N/A | 0.7532 | 0.6856 | 0.9282 | 0.7887 | 0.8312 |
| av_sync_1 | AVSync | tsm.pt | audio_resnet18.pt | 0.3333 | 0.3333 | 1.0000 | 0.5000 | 0.0000 / NaN |
| concat_tsm | Concat | single_frame.pt | audio_resnet18.pt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 / NaN |
| cross_attention_single | CrossAttention | single_frame.pt | audio_resnet18.pt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 / NaN |
| fast_fusion | FastFusion | tsm.pt | audio_resnet18.pt | 0.6113 | 0.5618 | 0.6849 | 0.6173 | 0.7000 |