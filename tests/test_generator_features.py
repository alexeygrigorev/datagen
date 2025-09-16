"""Tests for feature generation functionality in DatasetGenerator."""

import numpy as np
import pandas as pd
import pytest
from datagen.generator import DatasetGenerator
from datagen.schemas import DatasetPlan, Feature


class TestFeatureGeneration:
    """Test the feature generation methods in DatasetGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal plan for testing
        self.plan = DatasetPlan(
            task="classification",
            name="Test Plan",
            description="Test dataset plan",
            dataset_name="test_dataset",
            features=[],  # Will be set per test
            target_name="target",
            target_formula="0.5*feature1",
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
        result = self.generator._generate_numerical_feature("normal(0,1)")
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # Check statistical properties (with reasonable tolerance for random data)
        assert -3 < np.mean(result) < 3  # Should be around 0
        assert 0.5 < np.std(result) < 2.0  # Should be around 1
    
    def test_generate_numerical_feature_uniform(self):
        """Test uniform distribution generation."""
        result = self.generator._generate_numerical_feature("uniform(0,10)")
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # All values should be in range [0, 10]
        assert np.all(result >= 0)
        assert np.all(result <= 10)
        # Should have reasonable spread
        assert np.max(result) - np.min(result) > 5
    
    def test_generate_numerical_feature_lognormal(self):
        """Test lognormal distribution generation."""
        result = self.generator._generate_numerical_feature("lognormal(0,0.5)")
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # Lognormal values should all be positive
        assert np.all(result > 0)
    
    def test_generate_numerical_feature_poisson(self):
        """Test poisson distribution generation."""
        result = self.generator._generate_numerical_feature("poisson(3)")
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # Poisson values should be non-negative integers (converted to float)
        assert np.all(result >= 0)
        # Mean should be roughly around lambda=3
        assert 1 < np.mean(result) < 6
    
    def test_generate_numerical_feature_unknown_distribution(self):
        """Test fallback behavior for unknown distributions."""
        result = self.generator._generate_numerical_feature("unknown_dist(1,2)")
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # Should fallback to normal(0,1)
        assert -4 < np.mean(result) < 4
    
    def test_generate_categorical_feature_string_list(self):
        """Test categorical feature with string list."""
        feature = Feature(
            name="test_cat",
            type="categorical",
            distribution="categorical",
            domain_semantics="test category",
            categories=["A", "B", "C"]
        )
        
        result = self.generator._generate_categorical_feature(feature)
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # All values should be from the specified categories
        unique_values = set(result)
        assert unique_values.issubset({"A", "B", "C"})
        # Should have multiple categories represented (with high probability)
        assert len(unique_values) > 1
    
    def test_generate_categorical_feature_no_categories(self):
        """Test categorical feature fallback when no categories specified."""
        feature = Feature(
            name="test_cat",
            type="categorical", 
            distribution="categorical",
            domain_semantics="test category",
            categories=None
        )
        
        result = self.generator._generate_categorical_feature(feature)
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # Should fallback to binary A/B
        unique_values = set(result)
        assert unique_values.issubset({"A", "B"})
    
    def test_generate_categorical_feature_empty_categories(self):
        """Test categorical feature fallback when empty categories list."""
        feature = Feature(
            name="test_cat",
            type="categorical",
            distribution="categorical", 
            domain_semantics="test category",
            categories=[]
        )
        
        result = self.generator._generate_categorical_feature(feature)
        
        assert len(result) == 100
        # Should fallback to binary A/B
        unique_values = set(result)
        assert unique_values.issubset({"A", "B"})
    
    def test_generate_binary_feature_bernoulli(self):
        """Test binary feature with bernoulli distribution."""
        result = self.generator._generate_binary_feature("bernoulli(0.7)")
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # Values should be 0 or 1
        assert set(result).issubset({0, 1})
        # With p=0.7, should have more 1s than 0s (probabilistically)
        assert 0.4 < np.mean(result) < 1.0  # Allow for random variation
    
    def test_generate_binary_feature_default_probability(self):
        """Test binary feature with default probability (0.5)."""
        result = self.generator._generate_binary_feature("uniform(0,1)")  # Non-bernoulli dist
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        # Values should be 0 or 1
        assert set(result).issubset({0, 1})
        # Should use default p=0.5
        assert 0.2 < np.mean(result) < 0.8  # Allow for random variation
    
    def test_generate_ordinal_feature(self):
        """Test ordinal feature generation (uses categorical logic)."""
        feature = Feature(
            name="test_ordinal",
            type="categorical",  # ordinal is treated as categorical
            distribution="ordinal",
            domain_semantics="test ordinal",
            categories=["low", "medium", "high"]
        )
        
        result = self.generator._generate_ordinal_feature(feature)
        
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        unique_values = set(result)
        assert unique_values.issubset({"low", "medium", "high"})
    
    def test_generate_datetime_feature(self):
        """Test datetime feature generation."""
        result = self.generator._generate_datetime_feature("datetime")
        
        assert len(result) == 100
        # Result is a pandas Index with date strings, not a numpy array
        assert hasattr(result, '__len__')
        # All values should be valid date strings in 2023
        for date_str in result[:5]:  # Check first 5
            assert "2023" in date_str
            # Should be parseable as date
            pd.to_datetime(date_str)
    
    def test_generate_feature_dispatcher(self):
        """Test the main _generate_feature method dispatches correctly."""
        # Test numerical
        numerical_feature = Feature(
            name="num_test",
            type="numerical",
            distribution="normal(0,1)",
            domain_semantics="test numerical"
        )
        result = self.generator._generate_feature(numerical_feature)
        assert len(result) == 100
        assert isinstance(result, np.ndarray)
        
        # Test categorical
        categorical_feature = Feature(
            name="cat_test", 
            type="categorical",
            distribution="categorical",
            domain_semantics="test categorical",
            categories=["X", "Y"]
        )
        result = self.generator._generate_feature(categorical_feature)
        assert len(result) == 100
        assert set(result).issubset({"X", "Y"})
    
    def test_generate_feature_unsupported_type(self):
        """Test error handling for unsupported feature types."""
        unsupported_feature = Feature(
            name="unsupported",
            type="categorical",  # Will modify after creation
            distribution="test",
            domain_semantics="test"
        )
        # Hack to test unsupported type (since Literal restricts valid values)
        unsupported_feature.type = "unsupported_type"
        
        with pytest.raises(ValueError, match="Unsupported feature type"):
            self.generator._generate_feature(unsupported_feature)
    
    def test_parse_params_normal(self):
        """Test parameter parsing for normal distribution."""
        params = self.generator._parse_params("normal(0,1)", "normal")
        assert params == [0.0, 1.0]
    
    def test_parse_params_multiple_params(self):
        """Test parameter parsing with multiple parameters."""
        params = self.generator._parse_params("uniform(2.5,7.8)", "uniform")
        assert params == [2.5, 7.8]
    
    def test_parse_params_single_param(self):
        """Test parameter parsing with single parameter."""
        params = self.generator._parse_params("poisson(3.5)", "poisson")
        assert params == [3.5]
    
    def test_parse_params_invalid_format(self):
        """Test parameter parsing error handling."""
        with pytest.raises(ValueError, match="Cannot parse distribution"):
            self.generator._parse_params("invalid_format", "normal")
    
    def test_different_row_counts(self):
        """Test that feature generation respects different row counts."""
        # Test with different row count
        generator_small = DatasetGenerator(self.plan, 50, self.answers_dict)
        result = generator_small._generate_numerical_feature("normal(0,1)")
        assert len(result) == 50
        
        generator_large = DatasetGenerator(self.plan, 200, self.answers_dict)
        result = generator_large._generate_numerical_feature("normal(0,1)")
        assert len(result) == 200
    
    def test_random_seed_reproducibility(self):
        """Test that the same seed produces the same results."""
        gen1 = DatasetGenerator(self.plan, 100, self.answers_dict)
        gen2 = DatasetGenerator(self.plan, 100, self.answers_dict)
        
        result1 = gen1._generate_numerical_feature("normal(0,1)")
        result2 = gen2._generate_numerical_feature("normal(0,1)")
        
        # Should be identical with same seed
        np.testing.assert_array_equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__])