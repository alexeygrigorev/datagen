"""Tests for data quality features in DatasetGenerator."""

import numpy as np
import pandas as pd
from datagen.generator import DatasetGenerator
from datagen.schemas import (
    CategoricalFeature,
    DatasetPlan,
    LinearTerm,
    NormalDist,
    NumericalFeature,
    UniformDist,
)


class TestDataQuality:
    """Test the data quality methods in DatasetGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plan = DatasetPlan(
            task="classification",
            name="Data Quality Test Plan",
            description="Test dataset plan for data quality features",
            dataset_name="data_quality_test",
            features=[
                NumericalFeature(
                    name="feature1",
                    distribution=NormalDist(mean=50, std=10),
                    domain_semantics="test feature 1",
                    missing_rate=0.0,
                    rounding_precision=None
                ),
                NumericalFeature(
                    name="feature2",
                    distribution=UniformDist(low=0, high=100),
                    domain_semantics="test feature 2",
                    missing_rate=0.0,
                    rounding_precision=None
                )
            ],
            target_name="target",
            target_formula=[
                LinearTerm(feature="feature1", coefficient=0.5),
                LinearTerm(feature="feature2", coefficient=0.3),
            ],
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

        self.test_df = pd.DataFrame({
            'feature1': np.random.RandomState(42).normal(50, 10, 100),
            'feature2': np.random.RandomState(42).uniform(0, 100, 100),
            'feature3': np.random.RandomState(42).normal(0, 1, 100)
        })

    # ========== Rounding Tests ==========

    def test_apply_rounding_integer(self):
        self.plan.features[0].rounding_precision = "integer"

        result_df = self.generator._apply_rounding(self.test_df)

        assert result_df['feature1'].dtype.name == 'Int64'
        assert all(result_df['feature1'] == result_df['feature1'].round())

        pd.testing.assert_series_equal(result_df['feature2'], self.test_df['feature2'])
        pd.testing.assert_series_equal(result_df['feature3'], self.test_df['feature3'])

    def test_apply_rounding_decimal_places(self):
        test_cases = [
            ("0.1", 1),
            ("0.01", 2),
            ("0.001", 3)
        ]

        for precision_str, expected_decimals in test_cases:
            self.plan.features[0].rounding_precision = precision_str

            result_df = self.generator._apply_rounding(self.test_df)

            rounded_values = np.round(self.test_df['feature1'], expected_decimals)
            pd.testing.assert_series_equal(result_df['feature1'], rounded_values, check_names=False)

    def test_apply_rounding_nearest_values(self):
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

            expected = (np.round(self.test_df['feature1'] / nearest_value) * nearest_value).astype('Int64')
            pd.testing.assert_series_equal(result_df['feature1'], expected, check_names=False)

            assert all(result_df['feature1'] % nearest_value == 0)

    def test_apply_rounding_whole_numbers(self):
        self.plan.features[0].rounding_precision = "1"

        result_df = self.generator._apply_rounding(self.test_df)

        assert result_df['feature1'].dtype.name == 'Int64'
        expected = np.round(self.test_df['feature1'], 0).astype('Int64')
        pd.testing.assert_series_equal(result_df['feature1'], expected, check_names=False)

    def test_apply_rounding_unknown_precision(self):
        self.plan.features[0].rounding_precision = "unknown_precision"

        result_df = self.generator._apply_rounding(self.test_df)

        pd.testing.assert_series_equal(result_df['feature1'], self.test_df['feature1'])

    def test_apply_rounding_no_precision(self):
        self.plan.features[0].rounding_precision = None

        result_df = self.generator._apply_rounding(self.test_df)

        pd.testing.assert_frame_equal(result_df, self.test_df)

    def test_apply_rounding_non_numerical_features(self):
        """Test that rounding is not applied to non-numerical features."""
        categorical_feature = CategoricalFeature(
            name="feature3",
            categories=["A", "B", "C"],
            domain_semantics="test categorical",
        )
        self.plan.features.append(categorical_feature)

        test_df_with_cat = self.test_df.copy()
        test_df_with_cat['feature3'] = ['A', 'B', 'C'] * 33 + ['A']

        result_df = self.generator._apply_rounding(test_df_with_cat)

        pd.testing.assert_series_equal(result_df['feature3'], test_df_with_cat['feature3'])

    # ========== Missingness Tests ==========

    def test_apply_missingness_basic(self):
        self.plan.features[0].missing_rate = 0.2

        result_df = self.generator._apply_missingness(self.test_df)

        missing_rate = result_df['feature1'].isna().mean()
        assert 0.1 < missing_rate < 0.3

        assert result_df['feature2'].isna().sum() == 0
        assert result_df['feature3'].isna().sum() == 0

    def test_apply_missingness_zero_rate(self):
        self.plan.features[0].missing_rate = 0.0

        result_df = self.generator._apply_missingness(self.test_df)

        assert result_df.isna().sum().sum() == 0
        pd.testing.assert_frame_equal(result_df, self.test_df)

    def test_apply_missingness_high_rate(self):
        self.plan.features[0].missing_rate = 0.8

        result_df = self.generator._apply_missingness(self.test_df)

        missing_rate = result_df['feature1'].isna().mean()
        assert 0.7 < missing_rate < 0.9

    def test_apply_missingness_multiple_features(self):
        self.plan.features[0].missing_rate = 0.1
        self.plan.features[1].missing_rate = 0.3

        result_df = self.generator._apply_missingness(self.test_df)

        missing_rate_1 = result_df['feature1'].isna().mean()
        missing_rate_2 = result_df['feature2'].isna().mean()

        assert 0.05 < missing_rate_1 < 0.15
        assert 0.2 < missing_rate_2 < 0.4

        assert result_df['feature3'].isna().sum() == 0

    def test_apply_missingness_missing_column(self):
        missing_feature = NumericalFeature(
            name="missing_feature",
            distribution=NormalDist(mean=0, std=1),
            domain_semantics="missing feature",
            missing_rate=0.5
        )
        self.plan.features.append(missing_feature)

        result_df = self.generator._apply_missingness(self.test_df)

        pd.testing.assert_frame_equal(result_df, self.test_df)

    # ========== Outliers Tests ==========

    def test_apply_outliers_none(self):
        self.answers_dict['outliers'] = 'none'
        generator = DatasetGenerator(self.plan, 100, self.answers_dict)

        result_df = generator._apply_outliers(self.test_df)

        pd.testing.assert_frame_equal(result_df, self.test_df)

    def test_apply_outliers_slight(self):
        self.answers_dict['outliers'] = 'slight'
        generator = DatasetGenerator(self.plan, 100, self.answers_dict)

        original_df = self.test_df.copy()
        result_df = generator._apply_outliers(self.test_df)

        diff_feature1 = np.abs(result_df['feature1'] - original_df['feature1'])
        diff_feature2 = np.abs(result_df['feature2'] - original_df['feature2'])

        outliers_count = (diff_feature1 > 3 * original_df['feature1'].std()).sum()
        outliers_count += (diff_feature2 > 3 * original_df['feature2'].std()).sum()

        assert outliers_count > 0

    def test_get_outliers_rate(self):
        assert self.generator._get_outliers_rate("none") == 0.0
        assert self.generator._get_outliers_rate("slight") == 0.02
        assert self.generator._get_outliers_rate("unknown") == 0.0

    def test_inject_outliers_basic(self):
        test_df = self.test_df.copy()
        original_mean = test_df['feature1'].mean()
        original_std = test_df['feature1'].std()

        self.generator._inject_outliers(test_df, 'feature1', 0.1)

        changed_values = np.abs(test_df['feature1'] - self.test_df['feature1']) > 0
        assert 8 <= changed_values.sum() <= 12

        changed_indices = np.where(changed_values)[0]
        for idx in changed_indices:
            outlier_value = test_df.loc[idx, 'feature1']
            z_score = abs((outlier_value - original_mean) / original_std)
            assert z_score > 3.0

    def test_inject_outliers_zero_rate(self):
        test_df = self.test_df.copy()

        self.generator._inject_outliers(test_df, 'feature1', 0.0)

        pd.testing.assert_series_equal(test_df['feature1'], self.test_df['feature1'])

    def test_inject_outliers_small_dataset(self):
        small_df = self.test_df.head(5).copy()

        self.generator._inject_outliers(small_df, 'feature1', 0.1)

        pd.testing.assert_series_equal(small_df['feature1'], self.test_df.head(5)['feature1'])

    # ========== Integration Tests ==========

    def test_data_quality_pipeline_integration(self):
        self.plan.features[0].rounding_precision = "0.1"
        self.plan.features[0].missing_rate = 0.1
        self.plan.features[1].rounding_precision = "0.1"
        self.plan.features[1].missing_rate = 0.05

        self.answers_dict['outliers'] = 'none'
        generator = DatasetGenerator(self.plan, 100, self.answers_dict)

        df = self.test_df.copy()
        df = generator._apply_rounding(df)
        df = generator._apply_missingness(df)

        assert all(df['feature1'].dropna().round(1) == df['feature1'].dropna())
        assert all(df['feature2'].dropna().round(1) == df['feature2'].dropna())

        assert df['feature1'].isna().sum() > 0
        assert df['feature2'].isna().sum() >= 0

        assert df.shape == self.test_df.shape
