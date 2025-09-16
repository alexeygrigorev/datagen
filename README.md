# Synthetic ML Dataset Generator

A command-line tool that uses LLM-assisted specification to generate synthetic datasets for machine learning experiments.

## Features

- 🤖 **LLM-Powered**: Uses OpenAI's structured output to generate realistic dataset specifications
- 🎯 **Interactive Wizard**: Guided setup for dataset requirements
- 📊 **Multiple Domains**: Finance, healthcare, e-commerce, marketing, IoT, HR, or generic
- ⚡ **Fast Generation**: Efficient synthetic data creation with numpy/pandas
- 🔧 **CLI Ready**: Both interactive and non-interactive modes

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd datagen

uv sync

export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Interactive Mode (Recommended)

```bash
uv run python datagen.py
```

This launches an interactive wizard that will guide you through:
1. Dataset size (small/medium/large/very_large) 
2. Dataset description (free text like "customer churn prediction", "car fuel efficiency")
3. Random seed (optional)

The system automatically detects whether it's classification or regression from your description and lets the LLM determine the optimal number of features.

### Non-Interactive Mode

```bash
# Simple example
python syntheticgen.py \
  --size medium \
  --description "customer churn prediction for telecom company" \
  --seed 42 \
  --accept

# Another example  
python syntheticgen.py \
  --size small \
  --description "house prices based on location and features" \
  --task regression \
  --accept

# Use existing plan
python syntheticgen.py \
  --plan out/dataset_plan.json \
  --accept
```

All outputs are saved to the `out/` directory by default.

## Examples

### 1. Customer Churn Prediction

```bash
python syntheticgen.py --size medium --description "customer churn prediction for telecom company" --accept
```

**Generated Output:**
- ~10,000 rows with features like customer_tenure, monthly_charges, contract_type, etc.
- Binary classification target for churn prediction  
- Realistic telecom industry feature relationships

### 2. Financial Risk Prediction (Regression)

```bash
python datagen.py \
  --task regression \
  --size large \
  --features 20 \
  --domain finance \
  --seed 123 \
  --accept
```

**Generated Output:**
- 100,000 rows with financial features (income, credit_score, debt_ratio, etc.)
- Target variable representing risk score or loan amount
- Realistic feature relationships and distributions

### 4. Custom Dataset Description

```bash
# Interactive mode with custom description
python datagen.py
# Then select "Custom (describe your own)" and enter:
# "dataset about electric vehicle charging station usage patterns"
```

**Generated Output:**
- Features like station_location, charging_duration, battery_capacity, time_of_day
- Domain-specific value ranges and realistic correlations
- Tailored feature relationships for EV charging context

## Size Presets

| Preset | Base Rows | Variance | Actual Range | Default Features | Use Case |
|--------|-----------|----------|--------------|------------------|----------|
| `small` | 1,000 | ±200 | 800-1,200 | 8 | Quick prototyping, testing |
| `medium` | 10,000 | ±2,000 | 8,000-12,000 | 20 | Development, experiments |
| `large` | 100,000 | ±20,000 | 80,000-120,000 | 40 | Training larger models |
| `very_large` | 1,000,000 | ±200,000 | 800,000-1,200,000 | 60 | Production-scale testing |

> **Note**: Row counts are randomized within the variance range using the provided seed for reproducibility.

## Supported Domains

- **finance**: Credit scores, transaction amounts, risk indicators
- **healthcare**: Patient metrics, treatment history, readmission risk  
- **ecommerce**: Customer behavior, purchase patterns, churn prediction
- **marketing**: Campaign metrics, conversion rates, engagement scores
- **iot**: Sensor data, device status, failure prediction
- **hr**: Employee metrics, performance, attrition risk
- **generic**: General-purpose features for any ML task
- **custom**: Describe your own dataset requirements (e.g., "car fuel efficiency", "social media engagement")

## Output Files

Each generation creates:

1. **`dataset.csv`** - The synthetic dataset ready for ML
2. **`dataset_plan.json`** - Complete specification used by LLM
3. **`dataset_report.json`** - Statistics and data quality metrics
4. **`datagen.log`** - Detailed logs of the generation process

## Feature Types & Distributions

The LLM automatically selects appropriate distributions:

- **Numerical**: `normal(μ,σ)`, `uniform(a,b)`, `lognormal(μ,σ)`, `poisson(λ)`
- **Categorical**: Custom categories with specified probabilities
- **Binary**: `bernoulli(p)` for 0/1 features

## CLI Options

```bash
python syntheticgen.py [OPTIONS]

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
# 1. Generate a medium e-commerce dataset
python datagen.py --task classification --size medium --features 15 --domain ecommerce --seed 42 --accept

# 2. Check the output
ls -la
# dataset.csv (8K-12K rows, 16 columns including target)
# dataset_plan.json (LLM specification)
# dataset_report.json (statistics)

# 3. Generate custom dataset interactively
python datagen.py
# Select "Custom" domain and describe: "restaurant review sentiment analysis"

# 4. Load in Python for ML
import pandas as pd
df = pd.read_csv('dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Ready for sklearn, etc.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
