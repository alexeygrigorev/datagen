"""Tests for structured target formula evaluation (no string parsing)."""

import numpy as np
import pandas as pd
from datagen.generator import DatasetGenerator
from datagen.schemas import (
    CategoricalTerm,
    ConstantTerm,
    DatasetPlan,
    LinearTerm,
    NoiseTerm,
    NormalDist,
    NumericalFeature,
    UniformDist,
)


def make_generator(target_formula, seed=42):
    plan = DatasetPlan(
        task="regression",
        name="Test Plan",
        description="Test",
        dataset_name="test_dataset",
        features=[],
        target_name="target",
        target_formula=target_formula,
        domain="generic",
        seed=seed,
        rows=100,
    )
    return DatasetGenerator(plan, 100, {"task": "regression", "size": "small", "domain": "generic", "seed": seed})


class TestStructuredTargetTerms:
    def setup_method(self):
        self.df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5],
            'category': ['A', 'B', 'A']
        })

    def test_linear_terms(self):
        gen = make_generator([
            LinearTerm(feature='feature1', coefficient=2.0),
            LinearTerm(feature='feature2', coefficient=-0.5),
        ])
        result = gen._evaluate_formula(gen.plan.target_formula, self.df)
        expected = np.array([1.75, 3.25, 4.75])
        np.testing.assert_array_almost_equal(result, expected)

    def test_categorical_terms(self):
        gen = make_generator([
            CategoricalTerm(feature='category', value='A', coefficient=1.0),
            CategoricalTerm(feature='category', value='B', coefficient=-0.5),
        ])
        result = gen._evaluate_formula(gen.plan.target_formula, self.df)
        expected = np.array([1.0, -0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_constant_terms(self):
        gen = make_generator([
            ConstantTerm(value=2.5),
            ConstantTerm(value=-1.0),
        ])
        result = gen._evaluate_formula(gen.plan.target_formula, self.df)
        expected = np.array([1.5, 1.5, 1.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_noise_terms(self):
        gen = make_generator([
            NoiseTerm(distribution=NormalDist(mean=0, std=1), coefficient=1.0),
            NoiseTerm(distribution=UniformDist(low=0, high=2), coefficient=0.5),
        ])
        result = gen._evaluate_formula(gen.plan.target_formula, self.df)
        assert len(result) == 3
        assert not np.allclose(result, result[0])

    def test_mixed_terms(self):
        gen = make_generator([
            LinearTerm(feature='feature1', coefficient=1.0),
            CategoricalTerm(feature='category', value='A', coefficient=2.0),
            ConstantTerm(value=0.5),
        ])
        result = gen._evaluate_formula(gen.plan.target_formula, self.df)
        expected = np.array([3.5, 2.5, 5.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_missing_features_ignored(self):
        gen = make_generator([
            LinearTerm(feature='missing_feature', coefficient=1.0),
            CategoricalTerm(feature='missing_category', value='x', coefficient=1.0),
        ])
        result = gen._evaluate_formula(gen.plan.target_formula, self.df)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0, 0.0]))
