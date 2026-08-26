import torch
from torch.utils.data import DataLoader
from typing import Tuple
from src.utils.config import load_config
from src.models.model_factory import create_model
from src.datasets.audio_dataset import AudioDataset
from src.datasets.transforms import build_eval_transforms


def load_model_and_dataset(
    config_path: str,
    manifest_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, DataLoader, str]:
    """Load model and dataset based on config."""
    config = load_config(config_path)
    model_name = config["model"].get("name", "resnet18")
    
    # Determine modality
    if model_name == "audio_resnet18":
        modality = "audio"
        audio_config = config.get("data", {}).get("audio", {})
        dataset = AudioDataset(
            dataset_path=manifest_path,
            sample_rate=audio_config.get("sample_rate", 16000),
            n_fft=audio_config.get("n_fft", 400),
            hop_length=audio_config.get("hop_length", 160),
            n_mels=audio_config.get("n_mels", 80),
            f_min=audio_config.get("f_min", 0.0),
            f_max=audio_config.get("f_max", None),
            power=audio_config.get("power", 2.0),
            transform=None,
        )
    elif model_name in ["concat_fusion", "gated_fusion"]:
        modality = "fusion"
        # For fusion, we need both video and audio datasets
        # Get fusion config for projection settings
        fusion_config = config.get("fusion", {})
        use_projection = fusion_config.get("use_projection", False)
        
        # Create dataset for fusion - need both video and audio paths from manifest
        from src.datasets.multimodal_dataset import MultimodalDataset
        image_size = config["data"].get("image_size", 224)
        dataset = MultimodalDataset(
            manifest_path,
            transform=build_eval_transforms(image_size),  # Apply image transforms
            audio_transform=None,
            image_path_column="face_path",
            audio_path_column="audio_path",
            label_column="label",
            id_column="video_id",
            metadata_columns=None,
            root_dir=None,
        )
    else:
        modality = "video"
        use_temporal = config["model"].get("temporal", {}).get("enabled", False)
        image_size = config["data"].get("image_size", 224)
        
        if use_temporal:
            from src.datasets.multi_frame_dataset import MultiFrameDataset
            dataset = MultiFrameDataset(
                manifest_path,
                transform=build_eval_transforms(image_size),
                sequence_length=config["model"]["temporal"].get("sequence_length", 32),
                stride=config["model"]["temporal"].get("stride", 32),
            )
        else:
            from src.datasets.dataset import PreparedDataset
            dataset = PreparedDataset(
                manifest_path, transform=build_eval_transforms(image_size)
            )
    
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create and load model
    model = create_model(config["model"], num_classes=config["model"].get("num_classes", 2)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, loader, modality
