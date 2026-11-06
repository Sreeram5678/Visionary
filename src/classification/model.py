"""
Fine-Grained Vehicle Classification: Deep Learning Model Architecture.

Implements ConvNeXt-Tiny and EfficientNet backbones with penultimate
feature extraction tap for hybrid decision fusion.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import timm


class VehicleClassifier(nn.Module):
    """
    Fine-grained deep learning vehicle classifier with dual-output capability:
    - 196-class classification logits
    - Penultimate feature embedding tap for hybrid classical-DL fusion
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        num_classes: int = 196,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        """
        Initializes the VehicleClassifier.
        
        Args:
            backbone_name: Architecture name from timm ('convnext_tiny', 'efficientnet_b3').
            num_classes: Number of target categories (Stanford Cars: 196).
            pretrained: Whether to load ImageNet pretrained backbone weights.
            dropout_rate: Dropout probability before the final classification head.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # Instantiate feature extractor with pooled output (num_classes=0)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Automatically determine penultimate embedding dimension
        self.embedding_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through backbone and classification head.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            return_embedding: If True, returns (logits, embedding).
            
        Returns:
            Logits tensor (B, num_classes), or tuple (logits, embedding).
        """
        # Extract penultimate pooled embedding [B, embedding_dim]
        embedding = self.backbone(x)
        
        # Classification head
        logits = self.classifier(self.dropout(embedding))

        if return_embedding:
            return logits, embedding
        return logits

    def load_weights(self, weights_path: str, device: Union[str, torch.device] = "cpu") -> bool:
        """
        Loads fine-tuned model checkpoint from disk.
        
        Args:
            weights_path: Path to .pt or .pth checkpoint file.
            device: Target device string or torch.device.
            
        Returns:
            True if weights successfully loaded, False otherwise.
        """
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Strip possible 'module.' prefix from DataParallel training
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_key = k[7:] if k.startswith("module.") else k
                cleaned_state_dict[new_key] = v

            # Notebook checkpoints may name the head "fc" instead of "classifier".
            if "fc.weight" in cleaned_state_dict and "classifier.weight" not in cleaned_state_dict:
                cleaned_state_dict["classifier.weight"] = cleaned_state_dict.pop("fc.weight")
                cleaned_state_dict["classifier.bias"] = cleaned_state_dict.pop("fc.bias")

            # strict=True: a silent partial load would leave a random head and
            # produce confident garbage, so any key mismatch must fail loudly.
            self.load_state_dict(cleaned_state_dict, strict=True)
            self.eval()
            return True
        except Exception as e:
            raise RuntimeError(f"Could not load weights from {weights_path}: {e}") from e
