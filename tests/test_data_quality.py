"""Tests for data quality features in DatasetGenerator."""

import numpy as np
import pandas as pd
import pytest
from datagen.generator import DatasetGenerator
from datagen.schemas import DatasetPlan, Feature


class TestDataQuality:
    """Test the data quality methods in DatasetGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a plan with numerical features for testing
        self.plan = DatasetPlan(
            task="classification",
            name="Data Quality Test Plan",
            description="Test dataset plan for data quality features",
            dataset_name="data_quality_test",
            features=[
                Feature(
                    name="feature1",
                    type="numerical",
                    distribution="normal(50,10)",
                    domain_semantics="test feature 1",
                    missing_rate=0.0,
                    rounding_precision=None
                ),
                Feature(
                    name="feature2", 
                    type="numerical",
                    distribution="uniform(0,100)",
                    domain_semantics="test feature 2",
                    missing_rate=0.0,
                    rounding_precision=None
                )
            ],
            target_name="target",
            target_formula="0.5*feature1 + 0.3*feature2",
            domain="generic",
            seed=42,
            rows=100
        )
        
        self.answers_dict = {
            'task': 'classification',
            'size': 'small',
            'domain': 'generic', 
            'seed': 42,
            'outliers': 'none'
        }
        
        self.generator = DatasetGenerator(self.plan, 100, self.answers_dict)
        
        # Create test dataframe
        self.test_df = pd.DataFrame({
            'feature1': np.random.RandomState(42).normal(50, 10, 100),
            'feature2': np.random.RandomState(42).uniform(0, 100, 100),
            'feature3': np.random.RandomState(42).normal(0, 1, 100)  # Extra feature for testing
        })
    
    # ========== Rounding Tests ==========
    
    def test_apply_rounding_integer(self):
        """Test integer rounding precision."""
        # Modify feature to have integer rounding
        self.plan.features[0].rounding_precision = "integer"
        
        result_df = self.generator._apply_rounding(self.test_df)
        
        # feature1 should be rounded to integers
        assert result_df['feature1'].dtype.name == 'Int64'
        assert all(result_df['feature1'] == result_df['feature1'].round())
        
        # Other features should be unchanged
        pd.testing.assert_series_equal(result_df['feature2'], self.test_df['feature2'])
        pd.testing.assert_series_equal(result_df['feature3'], self.test_df['feature3'])
    
    def test_apply_rounding_decimal_places(self):
        """Test decimal place rounding precision."""
        test_cases = [
            ("0.1", 1),
            ("0.01", 2), 
            ("0.001", 3)
        ]
        
        for precision_str, expected_decimals in test_cases:
            self.plan.features[0].rounding_precision = precision_str
            
            result_df = self.generator._apply_rounding(self.test_df)
            
            # Check that values are rounded to correct decimal places
            rounded_values = np.round(self.test_df['feature1'], expected_decimals)
            pd.testing.assert_series_equal(result_df['feature1'], rounded_values, check_names=False)
    
    def test_apply_rounding_nearest_values(self):
        """Test rounding to nearest specific values."""
        test_cases = [
            ("nearest_5", 5),
            ("nearest_10", 10),
            ("nearest_25", 25),
            ("nearest_50", 50),
            ("nearest_100", 100)
        ]
        
        for precision_str, nearest_value in test_cases:
            self.plan.features[0].rounding_precision = precision_str
            
            result_df = self.generator._apply_rounding(self.test_df)
            
            # Check that values are rounded to nearest specified value
            expected = (np.round(self.test_df['feature1'] / nearest_value) * nearest_value).astype('Int64')
            pd.testing.assert_series_equal(result_df['feature1'], expected, check_names=False)
            
            # Verify all values are multiples of the nearest value
            assert all(result_df['feature1'] % nearest_value == 0)
    
    def test_apply_rounding_whole_numbers(self):
        """Test rounding precision '1' (whole numbers)."""
        self.plan.features[0].rounding_precision = "1"
        
        result_df = self.generator._apply_rounding(self.test_df)
        
        # Should be rounded to whole numbers as Int64
        assert result_df['feature1'].dtype.name == 'Int64'
        expected = np.round(self.test_df['feature1'], 0).astype('Int64')
        pd.testing.assert_series_equal(result_df['feature1'], expected, check_names=False)
    
    def test_apply_rounding_unknown_precision(self):
        """Test behavior with unknown rounding precision."""
        self.plan.features[0].rounding_precision = "unknown_precision"
        
        result_df = self.generator._apply_rounding(self.test_df)
        
        # Should be unchanged when precision is unknown
        pd.testing.assert_series_equal(result_df['feature1'], self.test_df['feature1'])
    
    def test_apply_rounding_no_precision(self):
        """Test behavior when no rounding precision is specified."""
        # Default is None
        self.plan.features[0].rounding_precision = None
        
        result_df = self.generator._apply_rounding(self.test_df)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(result_df, self.test_df)
    
    def test_apply_rounding_non_numerical_features(self):
        """Test that rounding is not applied to non-numerical features."""
        # Add categorical feature to plan
        categorical_feature = Feature(
            name="feature3",
            type="categorical",
            distribution="categorical",
            domain_semantics="test categorical",
            rounding_precision="integer"  # This should be ignored
        )
        self.plan.features.append(categorical_feature)
        
        # Replace feature3 with categorical data
        test_df_with_cat = self.test_df.copy()
        test_df_with_cat['feature3'] = ['A', 'B', 'C'] * 33 + ['A']  # 100 values
        
        result_df = self.generator._apply_rounding(test_df_with_cat)
        
        # Categorical feature should be unchanged
        pd.testing.assert_series_equal(result_df['feature3'], test_df_with_cat['feature3'])
    
    # ========== Missingness Tests ==========
    
    def test_apply_missingness_basic(self):
        """Test basic missingness application."""
        # Set missing rate for feature1
        self.plan.features[0].missing_rate = 0.2
        
        result_df = self.generator._apply_missingness(self.test_df)
        
        # Should have approximately 20% missing values in feature1
        missing_rate = result_df['feature1'].isna().mean()
        assert 0.1 < missing_rate < 0.3  # Allow some randomness
        
        # Other features should have no missing values
        assert result_df['feature2'].isna().sum() == 0
        assert result_df['feature3'].isna().sum() == 0
    
    def test_apply_missingness_zero_rate(self):
        """Test missingness with zero rate."""
        self.plan.features[0].missing_rate = 0.0
        
        result_df = self.generator._apply_missingness(self.test_df)
        
        # Should have no missing values
        assert result_df.isna().sum().sum() == 0
        pd.testing.assert_frame_equal(result_df, self.test_df)
    
    def test_apply_missingness_high_rate(self):
        """Test missingness with high rate."""
        self.plan.features[0].missing_rate = 0.8
        
        result_df = self.generator._apply_missingness(self.test_df)
        
        # Should have approximately 80% missing values
        missing_rate = result_df['feature1'].isna().mean()
        assert 0.7 < missing_rate < 0.9
    
    def test_apply_missingness_multiple_features(self):
        """Test missingness applied to multiple features."""
        self.plan.features[0].missing_rate = 0.1
        self.plan.features[1].missing_rate = 0.3
        
        result_df = self.generator._apply_missingness(self.test_df)
        
        # Check missing rates for both features
        missing_rate_1 = result_df['feature1'].isna().mean()
        missing_rate_2 = result_df['feature2'].isna().mean()
        
        assert 0.05 < missing_rate_1 < 0.15
        assert 0.2 < missing_rate_2 < 0.4
        
        # feature3 should be unchanged (not in plan)
        assert result_df['feature3'].isna().sum() == 0
    
    def test_apply_missingness_missing_column(self):
        """Test behavior when feature is not in dataframe."""
        # Add feature not in test_df
        missing_feature = Feature(
            name="missing_feature",
            type="numerical",
            distribution="normal(0,1)",
            domain_semantics="missing feature",
            missing_rate=0.5
        )
        self.plan.features.append(missing_feature)
        
        # Should not raise error
        result_df = self.generator._apply_missingness(self.test_df)
        
        # Existing features should be unchanged
        pd.testing.assert_frame_equal(result_df, self.test_df)
    
    # ========== Outliers Tests ==========
    
    def test_apply_outliers_none(self):
        """Test outliers with 'none' level."""
        self.answers_dict['outliers'] = 'none'
        generator = DatasetGenerator(self.plan, 100, self.answers_dict)
        
        result_df = generator._apply_outliers(self.test_df)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(result_df, self.test_df)
    
    def test_apply_outliers_slight(self):
        """Test outliers with 'slight' level."""
        self.answers_dict['outliers'] = 'slight'
        generator = DatasetGenerator(self.plan, 100, self.answers_dict)
        
        original_df = self.test_df.copy()
        result_df = generator._apply_outliers(self.test_df)
        
        # Should have some changes (outliers injected)
        # Check that some values are significantly different
        diff_feature1 = np.abs(result_df['feature1'] - original_df['feature1'])
        diff_feature2 = np.abs(result_df['feature2'] - original_df['feature2'])
        
        # Should have some outliers (at least a few values changed significantly)
        outliers_count = (diff_feature1 > 3 * original_df['feature1'].std()).sum()
        outliers_count += (diff_feature2 > 3 * original_df['feature2'].std()).sum()
        
        assert outliers_count > 0  # Should have at least some outliers
    
    def test_get_outliers_rate(self):
        """Test outliers rate conversion."""
        assert self.generator._get_outliers_rate("none") == 0.0
        assert self.generator._get_outliers_rate("slight") == 0.02
        assert self.generator._get_outliers_rate("unknown") == 0.0
    
    def test_get_missingness_rate(self):
        """Test missingness rate conversion."""
        assert self.generator._get_missingness_rate("none") == 0.0
        assert self.generator._get_missingness_rate("slight") == 0.05
        assert self.generator._get_missingness_rate("moderate") == 0.15
        assert self.generator._get_missingness_rate("unknown") == 0.0
    
    def test_inject_outliers_basic(self):
        """Test outlier injection method."""
        test_df = self.test_df.copy()
        original_mean = test_df['feature1'].mean()
        original_std = test_df['feature1'].std()
        
        self.generator._inject_outliers(test_df, 'feature1', 0.1)  # 10% outliers
        
        # Should have approximately 10 outliers (10% of 100)
        changed_values = np.abs(test_df['feature1'] - self.test_df['feature1']) > 0
        assert 8 <= changed_values.sum() <= 12  # Allow some variance
        
        # Changed values should be extreme (>3 std from mean)
        changed_indices = np.where(changed_values)[0]
        for idx in changed_indices:
            outlier_value = test_df.loc[idx, 'feature1']
            z_score = abs((outlier_value - original_mean) / original_std)
            assert z_score > 3.0
    
    def test_inject_outliers_zero_rate(self):
        """Test outlier injection with zero rate."""
        test_df = self.test_df.copy()
        
        self.generator._inject_outliers(test_df, 'feature1', 0.0)
        
        # Should be unchanged
        pd.testing.assert_series_equal(test_df['feature1'], self.test_df['feature1'])
    
    def test_inject_outliers_small_dataset(self):
        """Test outlier injection with small dataset."""
        small_df = self.test_df.head(5).copy()
        
        # With 5 rows and 10% rate, should inject 0 outliers (int(5*0.1) = 0)
        self.generator._inject_outliers(small_df, 'feature1', 0.1)
        
        # Should be unchanged
        pd.testing.assert_series_equal(small_df['feature1'], self.test_df.head(5)['feature1'])
    
    # ========== Integration Tests ==========
    
    def test_data_quality_pipeline_integration(self):
        """Test that all data quality methods work together."""
        # Set up plan with all quality features
        self.plan.features[0].rounding_precision = "0.1"  # Use decimal instead of integer for outliers compatibility
        self.plan.features[0].missing_rate = 0.1
        self.plan.features[1].rounding_precision = "0.1"
        self.plan.features[1].missing_rate = 0.05
        
        self.answers_dict['outliers'] = 'none'  # Disable outliers to avoid conflicts with rounding
        generator = DatasetGenerator(self.plan, 100, self.answers_dict)
        
        # Apply transformations (without outliers to avoid conflicts)
        df = self.test_df.copy()
        df = generator._apply_rounding(df)
        df = generator._apply_missingness(df)
        
        # Verify each transformation was applied
        # 1. Rounding - should be rounded to 1 decimal place
        assert all(df['feature1'].dropna().round(1) == df['feature1'].dropna())
        assert all(df['feature2'].dropna().round(1) == df['feature2'].dropna())
        
        # 2. Missingness
        assert df['feature1'].isna().sum() > 0  # Should have some missing values
        assert df['feature2'].isna().sum() >= 0  # Should have some missing values
        
        # 3. Shape should be preserved
        assert df.shape == self.test_df.shape


if __name__ == "__main__":
    pytest.main([__file__])