"""Tests for feature generation functionality in DatasetGenerator (structured)."""

import types

import numpy as np
import pandas as pd
from datagen.generator import DatasetGenerator
from datagen.schemas import (
    BinaryFeature,
    CategoricalFeature,
    DatasetPlan,
    LinearTerm,
    LogNormalDist,
    NormalDist,
    NumericalFeature,
    PoissonDist,
    UniformDist,
)


class TestFeatureGeneration:
    """Test the feature generation methods in DatasetGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plan = DatasetPlan(
            task="classification",
            name="Test Plan",
            description="Test dataset plan",
            dataset_name="test_dataset",
            features=[],
            target_name="target",
            target_formula=[LinearTerm(feature="feature1", coefficient=0.5)],
            domain="generic",
            seed=42,
            rows=100
        )

        self.answers_dict = {
            'task': 'classification',
            'size': 'small',
            'domain': 'generic',
            'seed': 42
        }

        self.generator = DatasetGenerator(self.plan, 100, self.answers_dict)

    def test_generate_numerical_feature_normal(self):
        """Test normal distribution generation."""
        result = self.generator._generate_numerical_feature(NormalDist(mean=0, std=1))

        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        assert -3 < np.mean(result) < 3
        assert 0.5 < np.std(result) < 2.0

    def test_generate_numerical_feature_uniform(self):
        """Test uniform distribution generation."""
        result = self.generator._generate_numerical_feature(UniformDist(low=0, high=10))

        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)
        assert np.all(result <= 10)
        assert np.max(result) - np.min(result) > 5

    def test_generate_numerical_feature_lognormal(self):
        """Test lognormal distribution generation."""
        result = self.generator._generate_numerical_feature(LogNormalDist(mean=0, sigma=0.5))

        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        assert np.all(result > 0)

    def test_generate_numerical_feature_poisson(self):
        """Test poisson distribution generation."""
        result = self.generator._generate_numerical_feature(PoissonDist(lam=3))

        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        assert np.all(result >= 0)
        assert 1 < np.mean(result) < 6

    def test_generate_categorical_feature_uniform(self):
        """Test categorical feature with uniform probabilities."""
        feature = CategoricalFeature(
            name="test_cat",
            categories=["A", "B", "C"],
            domain_semantics="test category",
        )

        result = self.generator._generate_categorical_feature(feature)

        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        unique_values = set(result)
        assert unique_values.issubset({"A", "B", "C"})
        assert len(unique_values) > 1

    def test_generate_categorical_feature_weighted(self):
        """Test categorical feature with explicit probabilities."""
        feature = CategoricalFeature(
            name="test_cat",
            categories=["A", "B"],
            probabilities=[0.9, 0.1],
            domain_semantics="test category",
        )

        result = self.generator._generate_categorical_feature(feature)

        assert len(result) == 100
        assert set(result).issubset({"A", "B"})
        # Heavily weighted toward A
        assert (result == "A").mean() > 0.7

    def test_generate_categorical_feature_no_categories(self):
        """Test categorical feature fallback when no categories specified."""
        feature = CategoricalFeature(
            name="test_cat",
            categories=[],
            domain_semantics="test category",
        )

        result = self.generator._generate_categorical_feature(feature)

        assert len(result) == 100
        assert set(result).issubset({"A", "B"})

    def test_generate_binary_feature(self):
        """Test binary feature with explicit p."""
        result = self.generator._generate_binary_feature(0.7)

        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        assert set(result).issubset({0, 1})
        assert 0.4 < np.mean(result) < 1.0

    def test_generate_binary_feature_default_probability(self):
        """Test binary feature with default probability (0.5)."""
        result = self.generator._generate_binary_feature()

        assert len(result) == 100
        assert set(result).issubset({0, 1})
        assert 0.2 < np.mean(result) < 0.8

    def test_generate_feature_dispatcher(self):
        """Test the main _generate_feature method dispatches correctly."""
        numerical_feature = NumericalFeature(
            name="num_test",
            distribution=NormalDist(mean=0, std=1),
            domain_semantics="test numerical"
        )
        result = self.generator._generate_feature(numerical_feature)
        assert len(result) == 100
        assert isinstance(result, np.ndarray)

        categorical_feature = CategoricalFeature(
            name="cat_test",
            categories=["X", "Y"],
            domain_semantics="test categorical",
        )
        result = self.generator._generate_feature(categorical_feature)
        assert len(result) == 100
        assert set(result).issubset({"X", "Y"})

        binary_feature = BinaryFeature(name="bin_test", p=0.5, domain_semantics="test binary")
        result = self.generator._generate_feature(binary_feature)
        assert len(result) == 100
        assert set(result).issubset({0, 1})

    def test_generate_feature_unsupported_type(self):
        """Test error handling for unsupported feature types."""
        unsupported = types.SimpleNamespace(type="unsupported_type", name="unsupported")

        try:
            self.generator._generate_feature(unsupported)
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "Unsupported feature type" in str(e)

    def test_different_row_counts(self):
        """Test that feature generation respects different row counts."""
        generator_small = DatasetGenerator(self.plan, 50, self.answers_dict)
        result = generator_small._generate_numerical_feature(NormalDist(mean=0, std=1))
        assert len(result) == 50

        generator_large = DatasetGenerator(self.plan, 200, self.answers_dict)
        result = generator_large._generate_numerical_feature(NormalDist(mean=0, std=1))
        assert len(result) == 200

    def test_random_seed_reproducibility(self):
        """Test that the same seed produces the same results."""
        gen1 = DatasetGenerator(self.plan, 100, self.answers_dict)
        gen2 = DatasetGenerator(self.plan, 100, self.answers_dict)

        result1 = gen1._generate_numerical_feature(NormalDist(mean=0, std=1))
        result2 = gen2._generate_numerical_feature(NormalDist(mean=0, std=1))

        np.testing.assert_array_equal(result1, result2)
