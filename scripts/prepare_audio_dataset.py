from __future__ import annotations

import argparse
from pathlib import Path


def main():
    """
    Prepare ASVspoof2021 DF dataset by loading a small subset and saving to disk.

    This script loads the SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF dataset
    using the Hugging Face datasets library and saves a small test split to disk
    for use with DeepGuard's audio deepfake detection pipeline.
    """
    from datasets import Dataset, load_dataset

    print("Loading ASVspoof2021_DF dataset subset...")
    
    # 1. Open the stream
    stream_dataset = load_dataset(
        "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
        split="test",  
        streaming=True
    ).take(5000)  # Take only the first 5000 samples

    # 2. Materialize the generator stream into a regular in-memory Dataset object
    # This downloads only your targeted 5,000 files instead of the huge shards
    print("Downloading and materializing 5,000 samples...")
    small_dataset = Dataset.from_generator(lambda: iter(stream_dataset))

    output_path = Path("./data/asvspoof_small")
    print(f"Saving dataset to {output_path}...")
    
    # 3. This will now work seamlessly without Pylance errors
    small_dataset.save_to_disk(str(output_path))

    print(f"\nDataset preparation complete!")
    print(f"  - Dataset size: {len(small_dataset)} samples")
    print(f"  - Saved to: {output_path}")
    print(f"  - Features: {small_dataset.column_names}")


if __name__ == "__main__":
    main()
