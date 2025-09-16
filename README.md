# Synthetic ML Dataset Generator

A command-line tool that uses LLM-assisted specification to generate synthetic datasets for machine learning experiments.

## Features

- 🤖 **LLM-Powered**: Uses OpenAI's structured output to generate realistic dataset specifications
- 🎯 **Interactive Wizard**: Guided setup for dataset requirements
- 📊 **Smart Rounding**: Intelligent precision based on feature semantics (integers for counts, decimals for measurements)
- 🏷️ **Custom Names**: LLM generates meaningful dataset names, user-confirmable
- 📂 **Organized Output**: Consistent naming across dataset, plan, and report files
- 🎲 **Missing Values**: Realistic missing data patterns per feature
- ⚡ **Fast Generation**: Efficient synthetic data creation with numpy/pandas
- 🔧 **CLI Ready**: Both interactive and non-interactive modes

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd datagen

# Install dependencies with uv
uv sync

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Interactive Mode (Recommended)

```bash
uv run python cli.py
```

This launches an interactive wizard that will guide you through:
1. Dataset size (small/medium/large/very_large) 
2. Dataset description (free text like "customer churn prediction", "car fuel efficiency")
3. Dataset filename confirmation (auto-generated from description)
4. Plan approval with complete feature list

The system automatically detects whether it's classification or regression from your description and lets the LLM determine the optimal number of features.

### Non-Interactive Mode

```bash
# Simple example
uv run python cli.py \
  --size medium \
  --description "customer churn prediction for telecom company" \
  --seed 42 \
  --accept

# Another example  
uv run python cli.py \
  --size small \
  --description "house prices based on location and features" \
  --task regression \
  --accept

# Use existing plan
uv run python cli.py \
  --plan out/house_prices_plan.json \
  --accept
```

All outputs are saved to the `out/` directory by default.

## Examples

### 1. Customer Churn Prediction

```bash
uv run python cli.py \
  --size medium \
  --description "customer churn prediction for telecom company" \
  --accept
```

**Generated Output:**
- ~10,000 rows with features like customer_tenure, monthly_charges, contract_type, etc.
- Binary classification target for churn prediction  
- Realistic telecom industry feature relationships
- Files: `customer_churn_prediction.csv`, `customer_churn_prediction_report.json`, `customer_churn_prediction_plan.json`

### 2. Car Specifications Dataset

```bash
uv run python cli.py \
  --task regression \
  --size small \
  --description "car fuel efficiency dataset" \
  --seed 123 \
  --accept
```

**Generated Output:**
- 1,000+ rows with features like engine_size (1 decimal), weight (nearest 10), cylinders (integer)
- Target variable: mpg (fuel efficiency)
- Smart rounding: engine_size=2.1, weight=2950, cylinders=6 (no .0 suffix)
- Files: `car_fuel_efficiency_dataset.csv`, etc.

### 3. Smartphone Specifications

```bash
# Interactive mode with custom description
uv run python cli.py
# Enter: "smartphone specifications with price prediction"
```

**Generated Output:**
- Features like screen_size (0.1 precision), ram_gb (integer), battery_mah (nearest 100)
- Target: price with realistic correlations
- Intelligent rounding based on real-world measurement precision

## Smart Rounding System

The LLM automatically assigns appropriate rounding precision based on feature semantics:

| Precision | Example Features | Output Format |
|-----------|------------------|---------------|
| `"integer"` | cylinders, gears, passengers | `4, 6, 8` (Int64 type) |
| `"1"` | horsepower, age, year | `150, 225, 180` (Int64 type) |
| `"0.1"` | fuel_efficiency, GPA, ratings | `24.5, 3.7, 4.2` |
| `"0.01"` | prices, percentages | `29.99, 15.75, 82.50` |
| `"nearest_10"` | weight, engine_displacement | `2950, 3180, 2640` (Int64) |
| `"nearest_100"` | battery_capacity, large_measurements | `3200, 4500, 3800` (Int64) |

## Size Presets

| Preset | Base Rows | Variance | Actual Range | Use Case |
|--------|-----------|----------|--------------|----------|
| `small` | 1,000 | ±200 | 800-1,200 | Quick prototyping, testing |
| `medium` | 10,000 | ±2,000 | 8,000-12,000 | Development, experiments |
| `large` | 100,000 | ±20,000 | 80,000-120,000 | Training larger models |
| `very_large` | 1,000,000 | ±200,000 | 800,000-1,200,000 | Production-scale testing |

> **Note**: Row counts are randomized within the variance range using the provided seed for reproducibility.

## Dataset Naming

- **Auto-Generated**: LLM creates file-safe names like `customer_churn_prediction`, `car_fuel_efficiency`
- **User Confirmation**: Interactive prompt to modify proposed name
- **Consistent Files**: All related files use the same base name:
  - `{dataset_name}.csv` (dataset)
  - `{dataset_name}_report.json` (statistics)  
  - `{dataset_name}_plan.json` (specification)

## Output Files

Each generation creates:

1. **`{dataset_name}.csv`** - The synthetic dataset ready for ML (with proper integer types in memory)
2. **`{dataset_name}_plan.json`** - Complete specification used by LLM including rounding rules
3. **`{dataset_name}_report.json`** - Statistics and data quality metrics
4. **`syntheticgen.log`** - Detailed logs of the generation process

## Feature Types & Distributions

The LLM automatically selects appropriate distributions and rounding:

- **Numerical**: `normal(μ,σ)`, `uniform(a,b)`, `lognormal(μ,σ)`, `poisson(λ)` with smart rounding
- **Categorical**: Custom categories with specified probabilities
- **Binary**: `bernoulli(p)` for 0/1 features
- **Missing Values**: Per-feature missing_rate (0.0 to 0.3) for realistic data quality

## CLI Options

```bash
uv run python cli.py [OPTIONS]

Options:
  --size [small|medium|large|very_large]  Dataset size preset
  --description TEXT                      Dataset description (free text)
  --task [classification|regression]      Task type (auto-detected if omitted)
  --seed INTEGER                          Random seed for reproducibility
  --plan PATH                             Path to existing dataset plan JSON file
  --accept                                Skip confirmation prompts
  --outdir PATH                           Output directory (default: out)
  --help                                  Show this message and exit
```

## Example Workflow

```bash
# 1. Generate a smartphone dataset with smart rounding
uv run python cli.py \
  --size medium \
  --description "smartphone specifications" \
  --accept

# 2. Check the output (example: smartphone_specifications.csv)
ls -la out/
# smartphone_specifications.csv (10K rows with proper types)
# smartphone_specifications_plan.json (LLM specification with rounding rules)
# smartphone_specifications_report.json (statistics)

# 3. Generate custom dataset interactively
uv run python cli.py
# Describe: "restaurant review sentiment analysis"
# Confirm filename: "restaurant_review_sentiment" ✓

# 4. Use existing plan
uv run python cli.py \
  --plan out/smartphone_specifications_plan.json \
  --accept

# 5. Load in Python for ML
import pandas as pd
df = pd.read_csv('out/smartphone_specifications.csv')

# Notice proper data types and realistic values:
# ram_gb: 4, 6, 8, 12 (integers, no .0)
# screen_size: 5.5, 6.1, 6.7 (1 decimal)
# price: 299.99, 599.00, 899.50 (2 decimals)

X = df.drop('price', axis=1)  # or whatever the target name is
y = df['price']

# Ready for sklearn, etc.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```