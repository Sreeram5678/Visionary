"""
Stanford Cars 196 Class Taxonomy and Metadata Parser.

Provides class index mappings, full vehicle names, and semantic decomposition
into Make, Model, Year, and Body Type.
"""

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Multi-word makes that require exact prefix matching
KNOWN_MULTIWORD_MAKES = [
    "AM General",
    "Aston Martin",
    "Land Rover",
    "Mercedes-Benz",
    "Rolls-Royce",
]

BODY_STYLES = [
    "Convertible",
    "Coupe",
    "Sedan",
    "Hatchback",
    "SUV",
    "Wagon",
    "Minivan",
    "Van",
    "Extended Cab",
    "Regular Cab",
    "SuperCab",
    "Crew Cab",
    "Cab",
]


@dataclass(frozen=True)
class CarMetadata:
    """Structured vehicle semantic attributes."""
    class_id: int
    full_name: str
    make: str
    model: str
    year: int
    body_style: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "full_name": self.full_name,
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "body_style": self.body_style,
        }


def _load_classes() -> List[str]:
    """Loads the 196 Stanford Cars class names from JSON."""
    json_path = os.path.join(os.path.dirname(__file__), "stanford_cars_196.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Missing Stanford Cars taxonomy file at {json_path}")


STANFORD_CARS_CLASSES: List[str] = _load_classes()
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(STANFORD_CARS_CLASSES)}


def parse_car_name(label: str, class_id: int = -1) -> CarMetadata:
    """
    Parses a Stanford Cars label string (e.g. 'Audi R8 Coupe 2012') into
    structured attributes (Make, Model, Year, Body Style).
    
    Args:
        label: Stanford Cars class string.
        class_id: Optional integer class index.
        
    Returns:
        CarMetadata instance.
    """
    label = label.strip()
    tokens = label.split()
    if not tokens:
        return CarMetadata(
            class_id=class_id,
            full_name=label,
            make="Unknown",
            model="Unknown",
            year=0,
            body_style="Unknown",
        )

    # 1. Extract Year (always the last 4-digit token in Stanford Cars)
    year = 0
    if tokens[-1].isdigit() and len(tokens[-1]) == 4:
        year = int(tokens[-1])
        tokens_without_year = tokens[:-1]
    else:
        tokens_without_year = tokens

    remaining_str = " ".join(tokens_without_year)

    # 2. Extract Make
    make = ""
    for multi in KNOWN_MULTIWORD_MAKES:
        if remaining_str.startswith(multi):
            make = multi
            model_str = remaining_str[len(multi):].strip()
            break

    if not make:
        # Default single word make
        make = tokens_without_year[0]
        model_str = " ".join(tokens_without_year[1:]).strip()

    # 3. Extract Body Style
    body_style = "Unknown"
    for style in BODY_STYLES:
        if re.search(rf"\b{re.escape(style)}\b", label, re.IGNORECASE):
            body_style = style
            break

    return CarMetadata(
        class_id=class_id if class_id >= 0 else CLASS_TO_IDX.get(label, -1),
        full_name=label,
        make=make,
        model=model_str if model_str else "Standard",
        year=year,
        body_style=body_style,
    )


# Precompute metadata for all 196 classes
CAR_METADATA_REGISTRY: List[CarMetadata] = [
    parse_car_name(name, idx) for idx, name in enumerate(STANFORD_CARS_CLASSES)
]


def get_class_name(idx: int) -> str:
    """Returns the full name string for a given class index (0..195)."""
    if 0 <= idx < len(STANFORD_CARS_CLASSES):
        return STANFORD_CARS_CLASSES[idx]
    raise IndexError(f"Class index {idx} out of range [0, {len(STANFORD_CARS_CLASSES)-1}]")


def get_metadata(idx: int) -> CarMetadata:
    """Returns structured CarMetadata for a given class index (0..195)."""
    if 0 <= idx < len(CAR_METADATA_REGISTRY):
        return CAR_METADATA_REGISTRY[idx]
    raise IndexError(f"Class index {idx} out of range [0, {len(CAR_METADATA_REGISTRY)-1}]")


def get_all_makes() -> List[str]:
    """Returns sorted unique list of all automotive manufacturers in the dataset."""
    return sorted(list(set(m.make for m in CAR_METADATA_REGISTRY)))


def get_classes_by_make(make: str) -> List[CarMetadata]:
    """Returns all vehicle classes belonging to a specified manufacturer."""
    return [m for m in CAR_METADATA_REGISTRY if m.make.lower() == make.lower()]
