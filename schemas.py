from typing import List, Optional, Literal
import random
from pydantic import BaseModel

# Type aliases using literals
TaskType = Literal["classification", "regression"]
FeatureType = Literal["numerical", "categorical", "binary"]
Domain = Literal["finance", "healthcare", "ecommerce", "marketing", "iot", "hr", "generic"]
SizePreset = Literal["small", "medium", "large", "very_large"]


class Feature(BaseModel):
    name: str
    type: FeatureType
    distribution: str
    domain_semantics: str
    categories: Optional[List[str]] = None
    missing_rate: Optional[float] = 0.0  # Probability of missing values (0.0 to 1.0)
    rounding_precision: Optional[str] = None  # e.g., "integer", "1", "0.1", "0.01", "nearest_5", "nearest_10"


class DatasetPlan(BaseModel):
    task: TaskType
    name: str
    description: str
    dataset_name: str  # File-safe name for saving (underscores instead of spaces)
    features: List[Feature]
    target_name: str  # Name of the target variable
    target_formula: str
    domain: Domain
    seed: int


class WizardAnswers(BaseModel):
    task: TaskType
    size: SizePreset
    features: Optional[int] = None
    domain: Domain
    custom_description: Optional[str] = None
    seed: int
    outdir: str = "."
    accept: bool = False


# Size presets mapping with randomness
SIZE_PRESETS = {
    "small": {"rows_base": 1000, "rows_variance": 200, "default_features": 8},
    "medium": {"rows_base": 10000, "rows_variance": 2000, "default_features": 20},
    "large": {"rows_base": 100000, "rows_variance": 20000, "default_features": 40},
    "very_large": {"rows_base": 1000000, "rows_variance": 200000, "default_features": 60},
}


def get_random_row_count(size: SizePreset, seed: int) -> int:
    """Get randomized row count for size preset."""
    preset = SIZE_PRESETS[size]
    rng = random.Random(seed)
    variance = rng.randint(-preset["rows_variance"], preset["rows_variance"])
    return max(100, preset["rows_base"] + variance)  # Minimum 100 rows