"""Simple test for row count in generated datasets."""

import tempfile
import json
from pathlib import Path

from datagen.schemas import DatasetPlan
from datagen.generator import DatasetGenerator


def test_dataset_generator_respects_row_count():
    """Test that DatasetGenerator produces the exact number of rows specified."""

    # Create a simple plan
    plan = DatasetPlan(
        task="classification",
        name="Row Count Test",
        description="Testing row count",
        dataset_name="row_count_test",
        features=[
            {
                "name": "feature1",
                "type": "numerical",
                "distribution": "normal(0,1)",
                "domain_semantics": "test feature",
                "missing_rate": 0.0,
                "rounding_precision": "0.1"
            }
        ],
        target_name="target",
        target_formula="0.5*feature1",
        domain="generic",
        seed=42,
        rows=1500  # Specific row count we want to test
    )
    
    # Create generator with specific row count
    answers_dict = {
        'task': 'classification',
        'size': 'medium',
        'domain': 'generic',
        'seed': 42
    }
    
    generator = DatasetGenerator(plan, 1500, answers_dict)
    df, report = generator.generate()
    
    # Verify the dataframe has exactly the number of rows we specified
    assert df.shape[0] == 1500, f"Expected 1500 rows, got {df.shape[0]}"
    assert df.shape[1] == 2, f"Expected 2 columns (feature + target), got {df.shape[1]}"
    
    # Verify the columns exist
    assert 'feature1' in df.columns
    assert 'target' in df.columns


def test_different_row_counts():
    """Test that different row counts produce different sized datasets."""
    
    base_plan = {
        "task": "regression",
        "name": "Variable Rows Test",
        "description": "Testing different row counts",
        "dataset_name": "variable_rows_test",
        "features": [
            {
                "name": "x",
                "type": "numerical",
                "distribution": "uniform(0,10)",
                "domain_semantics": "input variable",
                "missing_rate": 0.0,
                "rounding_precision": "0.01"
            }
        ],
        "target_name": "y",
        "target_formula": "2*x + 5",
        "domain": "generic",
        "seed": 123
    }
    
    answers_dict = {
        'task': 'regression',
        'size': 'small',
        'domain': 'generic',
        'seed': 123
    }
    
    test_cases = [100, 500, 1000, 2500]
    
    for expected_rows in test_cases:
        # Create plan with specific row count
        plan_dict = base_plan.copy()
        plan_dict['rows'] = expected_rows
        plan = DatasetPlan.model_validate(plan_dict)
        
        # Generate dataset
        generator = DatasetGenerator(plan, expected_rows, answers_dict)
        df, report = generator.generate()
        
        # Verify exact row count
        assert df.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df.shape[0]}"
        
        # Verify data quality
        assert not df.empty
        assert 'x' in df.columns
        assert 'y' in df.columns


def test_plan_with_rows_field_vs_without():
    """Test that plans with rows field use saved count, plans without use computed count."""
    
    # Plan WITH rows field
    plan_with_rows = DatasetPlan(
        task="classification",
        name="Plan With Rows",
        description="Plan that includes rows field",
        dataset_name="plan_with_rows",
        features=[
            {
                "name": "score",
                "type": "numerical", 
                "distribution": "normal(50,10)",
                "domain_semantics": "test score",
                "missing_rate": 0.0,
                "rounding_precision": "1"
            }
        ],
        target_name="pass",
        target_formula="score - 50",
        domain="generic",
        seed=999,
        rows=777  # Specific count
    )
    
    # Plan WITHOUT rows field (will be None)
    plan_without_rows = DatasetPlan(
        task="classification",
        name="Plan Without Rows",
        description="Plan without rows field",
        dataset_name="plan_without_rows",
        features=[
            {
                "name": "score",
                "type": "numerical",
                "distribution": "normal(50,10)",
                "domain_semantics": "test score",
                "missing_rate": 0.0,
                "rounding_precision": "1"
            }
        ],
        target_name="pass",
        target_formula="score - 50",
        domain="generic",
        seed=999
        # No rows field
    )
    
    answers = {'task': 'classification', 'size': 'small', 'domain': 'generic', 'seed': 999}
    
    # Test plan with rows field
    generator1 = DatasetGenerator(plan_with_rows, 777, answers)
    df1, _ = generator1.generate()
    assert df1.shape[0] == 777, f"Plan with rows=777 should produce 777 rows, got {df1.shape[0]}"
    
    # Test plan without rows field  
    generator2 = DatasetGenerator(plan_without_rows, 333, answers)  # Manually specify different count
    df2, _ = generator2.generate()
    assert df2.shape[0] == 333, f"Plan without rows should use provided count 333, got {df2.shape[0]}"
    
    # Verify they're different
    assert df1.shape[0] != df2.shape[0], "The two datasets should have different row counts"