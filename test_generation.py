#!/usr/bin/env python3

"""
Simple test script to validate dataset generation without questionary
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from schemas import DatasetPlan
from generator import generate_dataset

def test_generation():
    """Test dataset generation with the test plan."""
    
    # Load test plan
    with open('test_plan.json', 'r') as f:
        plan_data = json.load(f)
    
    plan = DatasetPlan.model_validate(plan_data)
    print(f"✅ Loaded plan: {plan.name}")
    print(f"   Features: {len(plan.features)}")
    print(f"   Target: {plan.target_name}")
    
    # Save plan to out directory
    Path("out").mkdir(exist_ok=True)
    plan_file = "out/dataset_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan.model_dump(), f, indent=2)
    
    # Generate dataset
    print("🎯 Generating dataset...")
    answers_dict = {"size": "small", "seed": plan.seed}
    dataset_file, report_file = generate_dataset(plan_file, answers_dict, "out")
    
    print(f"✅ Generated: {dataset_file}")
    print(f"✅ Report: {report_file}")
    
    # Show preview
    import pandas as pd
    df = pd.read_csv(dataset_file)
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"Target column: {plan.target_name}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    print(df.head(3))

if __name__ == "__main__":
    test_generation()