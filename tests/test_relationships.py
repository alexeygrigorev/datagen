"""Tests for domain-aware feature relationships and calibrated labels."""

import numpy as np
import pandas as pd
import pytest

from datagen.generator import DatasetGenerator
from datagen.schemas import (
    CategoricalFeature,
    ClassificationConfig,
    ConstantTerm,
    DatasetPlan,
    DerivedNumericalFeature,
    FeatureBounds,
    LinearTerm,
    MissingnessRule,
    NormalDist,
    NumericalFeature,
)


def make_plan(seed=42):
    return DatasetPlan(
        task="classification",
        name="Relationship test",
        description="Test dependency-aware generation",
        dataset_name="relationship_test",
        domain="marketing",
        seed=seed,
        rows=2000,
        features=[
            CategoricalFeature(
                name="segment",
                categories=["new", "returning"],
                probabilities=[0.6, 0.4],
            ),
            NumericalFeature(
                name="base_value",
                distribution=NormalDist(mean=50, std=8),
                bounds=FeatureBounds(low=0, high=100),
            ),
            DerivedNumericalFeature(
                name="engagement",
                formula=[
                    ConstantTerm(value=2),
                    LinearTerm(feature="base_value", coefficient=0.8),
                ],
                bounds=FeatureBounds(low=0, high=100),
            ),
        ],
        target_name="converted",
        target_formula=[
            ConstantTerm(value=0),
            LinearTerm(feature="engagement", coefficient=0.05),
        ],
        classification=ClassificationConfig(
            mode="bernoulli_logistic",
            target_rate=0.55,
            temperature=1.0,
        ),
        missingness_rules=[
            MissingnessRule(
                feature="engagement",
                rate=0.5,
                condition_feature="segment",
                condition_operator="eq",
                condition_value="new",
            )
        ],
    )


def make_answers(seed=42):
    return {"task": "classification", "size": "small", "seed": seed}


def test_dependency_relationships_bounds_and_conditional_missingness():
    generator = DatasetGenerator(make_plan(), 2000, make_answers())
    df, report = generator.generate()

    assert df["engagement"].dropna().between(0, 100).all()
    assert df["engagement"].corr(df["base_value"]) > 0.95
    assert df["converted"].isin([0, 1]).all()
    assert 0.45 < df["converted"].mean() < 0.65

    new_missing = df.loc[df["segment"] == "new", "engagement"].isna().mean()
    returning_missing = df.loc[df["segment"] == "returning", "engagement"].isna().mean()
    assert new_missing > 0.35
    assert returning_missing < 0.10
    assert report["data_quality"]["bound_violations"]["engagement"] == 0


def test_dependency_generation_is_reproducible():
    first, _ = DatasetGenerator(make_plan(), 2000, make_answers()).generate()
    second, _ = DatasetGenerator(make_plan(), 2000, make_answers()).generate()
    pd.testing.assert_frame_equal(first, second)


def test_derived_feature_must_reference_previous_columns():
    plan = make_plan()
    plan.features[-1].formula = [LinearTerm(feature="not_generated", coefficient=1)]
    with pytest.raises(ValueError, match="not been generated yet"):
        DatasetGenerator(plan, 2000, make_answers()).generate()
