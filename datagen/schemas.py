from typing import Annotated, List, Literal, Optional, Union
import random
from pydantic import BaseModel, Field

# Type aliases using literals
TaskType = Literal["classification", "regression"]
Domain = Literal[
    "finance",
    "healthcare",
    "ecommerce",
    "marketing",
    "automotive",
    "iot",
    "hr",
    "generic",
]
SizePreset = Literal["small", "medium", "large", "very_large"]


# ---------- Distributions (structured output replaces "normal(0,1)" strings) ----------

class NormalDist(BaseModel):
    kind: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = 1.0


class UniformDist(BaseModel):
    kind: Literal["uniform"] = "uniform"
    low: float = 0.0
    high: float = 1.0


class LogNormalDist(BaseModel):
    kind: Literal["lognormal"] = "lognormal"
    mean: float = 0.0
    sigma: float = 0.5


class PoissonDist(BaseModel):
    kind: Literal["poisson"] = "poisson"
    lam: float = 1.0


NumericalDistribution = Annotated[
    Union[NormalDist, UniformDist, LogNormalDist, PoissonDist],
    Field(discriminator="kind"),
]

NoiseDistribution = Annotated[
    Union[NormalDist, UniformDist],
    Field(discriminator="kind"),
]


class FeatureBounds(BaseModel):
    """Optional physical or business bounds for a generated numeric value."""

    low: Optional[float] = None
    high: Optional[float] = None


# ---------- Features (discriminated by type) ----------

class NumericalFeature(BaseModel):
    name: str
    type: Literal["numerical"] = "numerical"
    distribution: NumericalDistribution
    domain_semantics: str = ""
    missing_rate: float = 0.0
    rounding_precision: Optional[str] = None  # e.g. "integer", "1", "0.1", "0.01", "nearest_10"
    bounds: Optional[FeatureBounds] = None


class CategoricalFeature(BaseModel):
    name: str
    type: Literal["categorical"] = "categorical"
    categories: List[str]
    probabilities: Optional[List[float]] = None  # None = uniform; otherwise must match categories
    domain_semantics: str = ""
    missing_rate: float = 0.0


class BinaryFeature(BaseModel):
    name: str
    type: Literal["binary"] = "binary"
    p: float = 0.5  # P(x=1); replaces "bernoulli(p)" string
    domain_semantics: str = ""
    missing_rate: float = 0.0


Feature = Annotated[
    Union[NumericalFeature, CategoricalFeature, BinaryFeature],
    Field(discriminator="type"),
]


# ---------- Target formula terms (replaces formula string DSL) ----------

class LinearTerm(BaseModel):
    kind: Literal["linear"] = "linear"
    feature: str
    coefficient: float = 1.0


class CategoricalTerm(BaseModel):
    kind: Literal["categorical"] = "categorical"
    feature: str
    value: str
    coefficient: float = 1.0


class ConstantTerm(BaseModel):
    kind: Literal["constant"] = "constant"
    value: float = 0.0


class NoiseTerm(BaseModel):
    kind: Literal["noise"] = "noise"
    distribution: NoiseDistribution
    coefficient: float = 1.0


TargetTerm = Annotated[
    Union[LinearTerm, CategoricalTerm, ConstantTerm, NoiseTerm],
    Field(discriminator="kind"),
]


class DerivedNumericalFeature(BaseModel):
    """A numeric feature generated from already-generated features.

    Keeping this relationship in the plan makes the dependency structure explicit
    and reproducible instead of asking an LLM to imply correlations in prose.
    """

    name: str
    type: Literal["derived_numerical"] = "derived_numerical"
    formula: List[TargetTerm]
    domain_semantics: str = ""
    missing_rate: float = 0.0
    rounding_precision: Optional[str] = None
    bounds: Optional[FeatureBounds] = None


# Rebind the union after the target terms and derived feature are defined. The
# DatasetPlan below sees this complete union; existing plans remain valid.
Feature = Annotated[
    Union[NumericalFeature, CategoricalFeature, BinaryFeature, DerivedNumericalFeature],
    Field(discriminator="type"),
]


class MissingnessRule(BaseModel):
    """Domain-aware missingness, optionally limited to matching rows."""

    feature: str
    rate: float
    condition_feature: Optional[str] = None
    condition_operator: Optional[Literal["eq", "ne", "lt", "le", "gt", "ge"]] = None
    condition_value: Optional[Union[float, str]] = None


class ClassificationConfig(BaseModel):
    """How a classification score becomes an observed target."""

    mode: Literal["threshold", "bernoulli_logistic"] = "threshold"
    temperature: float = 1.0
    target_rate: Optional[float] = None
    probability_clip_low: float = 0.01
    probability_clip_high: float = 0.99


def format_target_formula(terms: List[TargetTerm]) -> str:
    """Human-readable rendering of structured target terms (for CLI display)."""
    parts = []
    for t in terms:
        if isinstance(t, LinearTerm):
            parts.append(f"{t.coefficient}*{t.feature}")
        elif isinstance(t, CategoricalTerm):
            parts.append(f"{t.coefficient}*[{t.feature}=={t.value}]")
        elif isinstance(t, ConstantTerm):
            parts.append(f"{t.value}")
        elif isinstance(t, NoiseTerm):
            d = t.distribution
            if isinstance(d, NormalDist):
                parts.append(f"{t.coefficient}*normal({d.mean},{d.std})")
            elif isinstance(d, UniformDist):
                parts.append(f"{t.coefficient}*uniform({d.low},{d.high})")
            else:
                parts.append(f"{t.coefficient}*noise")
    return " + ".join(parts).replace("+ -", "- ")


class DatasetPlan(BaseModel):
    plan_version: int = 2
    task: TaskType
    name: str
    description: str
    dataset_name: str  # File-safe name for saving (underscores instead of spaces)
    features: List[Feature]
    target_name: str  # Name of the target variable
    target_formula: List[TargetTerm]
    domain: Domain
    seed: int
    rows: Optional[int] = None  # Number of rows to generate (computed if None)
    missingness_rules: List[MissingnessRule] = Field(default_factory=list)
    classification: Optional[ClassificationConfig] = None
    target_rounding_precision: Optional[str] = None
    target_noise_scale: Optional[float] = None


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
