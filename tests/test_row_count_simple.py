"""Simple test for row count in generated datasets."""

from datagen.schemas import DatasetPlan, LinearTerm, NormalDist, NumericalFeature, UniformDist
from datagen.generator import DatasetGenerator


def test_dataset_generator_respects_row_count():
    """Test that DatasetGenerator produces the exact number of rows specified."""

    plan = DatasetPlan(
        task="classification",
        name="Row Count Test",
        description="Testing row count",
        dataset_name="row_count_test",
        features=[
            NumericalFeature(
                name="feature1",
                distribution=NormalDist(mean=0, std=1),
                domain_semantics="test feature",
                missing_rate=0.0,
                rounding_precision="0.1"
            )
        ],
        target_name="target",
        target_formula=[LinearTerm(feature="feature1", coefficient=0.5)],
        domain="generic",
        seed=42,
        rows=1500
    )

    answers_dict = {
        'task': 'classification',
        'size': 'medium',
        'domain': 'generic',
        'seed': 42
    }

    generator = DatasetGenerator(plan, 1500, answers_dict)
    df, report = generator.generate()

    assert df.shape[0] == 1500, f"Expected 1500 rows, got {df.shape[0]}"
    assert df.shape[1] == 2, f"Expected 2 columns (feature + target), got {df.shape[1]}"

    assert 'feature1' in df.columns
    assert 'target' in df.columns


def test_different_row_counts():
    """Test that different row counts produce different sized datasets."""

    base_kwargs = dict(
        task="regression",
        name="Variable Rows Test",
        description="Testing different row counts",
        dataset_name="variable_rows_test",
        features=[
            NumericalFeature(
                name="x",
                distribution=UniformDist(low=0, high=10),
                domain_semantics="input variable",
                missing_rate=0.0,
                rounding_precision="0.01"
            )
        ],
        target_name="y",
        target_formula=[
            LinearTerm(feature="x", coefficient=2.0),
        ],
        domain="generic",
        seed=123
    )

    answers_dict = {
        'task': 'regression',
        'size': 'small',
        'domain': 'generic',
        'seed': 123
    }

    test_cases = [100, 500, 1000, 2500]

    for expected_rows in test_cases:
        plan = DatasetPlan(**base_kwargs, rows=expected_rows)

        generator = DatasetGenerator(plan, expected_rows, answers_dict)
        df, report = generator.generate()

        assert df.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df.shape[0]}"

        assert not df.empty
        assert 'x' in df.columns
        assert 'y' in df.columns


def test_plan_with_rows_field_vs_without():
    """Test that plans with rows field use saved count, plans without use computed count."""

    plan_with_rows = DatasetPlan(
        task="classification",
        name="Plan With Rows",
        description="Plan that includes rows field",
        dataset_name="plan_with_rows",
        features=[
            NumericalFeature(
                name="score",
                distribution=NormalDist(mean=50, std=10),
                domain_semantics="test score",
                missing_rate=0.0,
                rounding_precision="1"
            )
        ],
        target_name="pass",
        target_formula=[LinearTerm(feature="score", coefficient=1.0)],
        domain="generic",
        seed=999,
        rows=777
    )

    plan_without_rows = DatasetPlan(
        task="classification",
        name="Plan Without Rows",
        description="Plan without rows field",
        dataset_name="plan_without_rows",
        features=[
            NumericalFeature(
                name="score",
                distribution=NormalDist(mean=50, std=10),
                domain_semantics="test score",
                missing_rate=0.0,
                rounding_precision="1"
            )
        ],
        target_name="pass",
        target_formula=[LinearTerm(feature="score", coefficient=1.0)],
        domain="generic",
        seed=999
    )

    answers = {'task': 'classification', 'size': 'small', 'domain': 'generic', 'seed': 999}

    generator1 = DatasetGenerator(plan_with_rows, 777, answers)
    df1, _ = generator1.generate()
    assert df1.shape[0] == 777, f"Plan with rows=777 should produce 777 rows, got {df1.shape[0]}"

    generator2 = DatasetGenerator(plan_without_rows, 333, answers)
    df2, _ = generator2.generate()
    assert df2.shape[0] == 333, f"Plan without rows should use provided count 333, got {df2.shape[0]}"

    assert df1.shape[0] != df2.shape[0]
