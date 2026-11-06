"""
Fine-Grained Vehicle Classification: Inference Engine.

Preprocesses cropped vehicle ROIs, executes calibrated neural network inference,
and computes Top-1 and Top-5 class probabilities alongside penultimate embeddings.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from src.classification.class_names import (
    CAR_METADATA_REGISTRY,
    CarMetadata,
    get_metadata,
)
from src.classification.model import VehicleClassifier


@dataclass
class CarPrediction:
    """Individual vehicle class prediction."""
    class_id: int
    class_name: str
    make: str
    model: str
    year: int
    body_style: str
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "body_style": self.body_style,
            "probability": round(self.probability, 4),
            "confidence_percent": round(self.probability * 100.0, 2),
        }


@dataclass
class ClassificationResult:
    """Consolidated classification output with Top-K predictions and deep embedding."""
    top1: CarPrediction
    top5: List[CarPrediction]
    embedding: np.ndarray
    inference_time_ms: float
    is_calibrated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top1": self.top1.to_dict(),
            "top5": [p.to_dict() for p in self.top5],
            "embedding_dim": int(self.embedding.shape[0]),
            "inference_time_ms": round(self.inference_time_ms, 2),
        }


class VehicleClassifierEngine:
    """
    Inference runner managing image transformation, device dispatch,
    and calibrated Softmax output for 196 Stanford Cars categories.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        backbone_name: str = "convnext_tiny",
        device: Optional[Union[str, torch.device]] = None,
        temperature: float = 1.0,
        img_size: int = 224,
    ):
        """
        Initializes the VehicleClassifierEngine.
        
        Args:
            weights_path: Path to fine-tuned model checkpoint file.
            backbone_name: Backbone architecture name.
            device: Explicit device ('cuda', 'mps', 'cpu') or None for auto-selection.
            temperature: Softmax calibration temperature (> 1.0 reduces overconfidence).
            img_size: Image input resolution for the neural network.
        """
        # Auto-select fastest available compute device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.temperature = max(1e-4, float(temperature))
        self.img_size = img_size

        # Preprocessing transform using standard ImageNet mean and std
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Instantiate model
        self.model = VehicleClassifier(
            backbone_name=backbone_name,
            num_classes=196,
            pretrained=True if weights_path is None else False,
        )

        if weights_path:
            self.model.load_weights(weights_path, device=self.device)

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        top_k: int = 5,
    ) -> ClassificationResult:
        """
        Runs inference on a vehicle image or ROI crop.
        
        Args:
            image_input: RGB NumPy array, PIL Image, or file path.
            top_k: Number of highest-confidence predictions to return.
            
        Returns:
            ClassificationResult containing Top-1 & Top-K predictions,
            probabilities, and penultimate layer embedding.
        """
        pil_image = self._to_pil_image(image_input)

        # Apply transforms and add batch dimension [1, 3, H, W]
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            logits, embedding = self.model(tensor, return_embedding=True)
            
            # Apply temperature scaling for calibrated probabilities
            calibrated_logits = logits / self.temperature
            probabilities = torch.softmax(calibrated_logits, dim=-1).squeeze(0).cpu().numpy()
            embedding_np = embedding.squeeze(0).cpu().numpy()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Retrieve top K class indices
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions: List[CarPrediction] = []
        for rank, idx in enumerate(top_indices):
            meta = get_metadata(int(idx))
            pred = CarPrediction(
                class_id=int(idx),
                class_name=meta.full_name,
                make=meta.make,
                model=meta.model,
                year=meta.year,
                body_style=meta.body_style,
                probability=float(probabilities[idx]),
            )
            predictions.append(pred)

        return ClassificationResult(
            top1=predictions[0],
            top5=predictions[:top_k],
            embedding=embedding_np,
            inference_time_ms=elapsed_ms,
            is_calibrated=(self.temperature != 1.0),
        )

    @staticmethod
    def _to_pil_image(image_input: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        """Converts input image to RGB PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
            return img
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input image array is empty.")
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)
            if image_input.ndim == 2:
                return Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                return Image.fromarray(image_input[:, :, :3])
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                return Image.fromarray(image_input)
            else:
                raise ValueError(f"Unsupported image array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
