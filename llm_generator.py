import logging
from openai import OpenAI
from schemas import DatasetPlan, WizardAnswers, get_random_row_count


logger = logging.getLogger(__name__)


def generate_dataset_plan(answers: WizardAnswers) -> DatasetPlan:
    """Generate dataset plan using OpenAI structured output."""
    
    client = OpenAI()
    prompt = build_prompt(answers)
    
    try:
        response = client.responses.parse(
            model='gpt-4o',
            input=[
                {
                    "role": "system", 
                    "content": "You are a synthetic dataset specification generator. Generate detailed, realistic dataset plans."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            text_format=DatasetPlan
        )
        
        plan = response.output[0].content[0].parsed
        logger.info("Successfully generated dataset plan with structured output")
        return plan
        
    except Exception as e:
        logger.error(f"LLM structured output failed: {e}")
        return generate_fallback_plan(answers)


def build_prompt(answers: WizardAnswers) -> str:
    """Build prompt for LLM based on wizard answers."""
    
    rows = get_random_row_count(answers.size, answers.seed)
    description = answers.custom_description or "generic dataset"
    
    prompt = f"""Generate a synthetic dataset specification:

TASK: {answers.task}
DESCRIPTION: {description}
ROWS: {rows:,}
SEED: {answers.seed}

Based on the description "{description}", create an appropriate number of features (typically 5-20) that are realistic and meaningful for this context.
Use snake_case for feature names.
Include a target_name field with an appropriate name for the target variable.
Create a dataset_name field with a file-safe name (lowercase, underscores instead of spaces, no special characters).
For target_formula, create a formula that combines features meaningfully.
Some features should have missing_rate > 0 (typically 0.05-0.15 for a few features) to make the dataset realistic.

Examples:
- Classification: "-2.0 + 0.5*age - 0.1*income + 0.3*risk_score"
- Regression: "50000 + 100*experience + 5000*education_level + normal(0,5000)"

DISTRIBUTION EXAMPLES:
- normal(mu, sigma): Gaussian
- uniform(a, b): Uniform between a and b  
- poisson(lambda): Count data
- bernoulli(p): Binary 0/1

MISSING VALUES:
- Set missing_rate between 0.0 and 0.3 for some features
- Most features should have missing_rate = 0.0
- A few features (2-4) can have missing_rate between 0.05-0.15

ROUNDING PRECISION:
- For numerical features, specify appropriate rounding_precision based on real-world context:
  * "integer" - for counts (cylinders, gears, passengers, bedrooms)
  * "1" - for whole numbers (horsepower, weight in lbs, age, year)
  * "0.1" - for one decimal place (fuel efficiency, GPA, ratings)
  * "0.01" - for two decimal places (prices, percentages, precise measurements)
  * "nearest_5" - round to nearest 5 (ages in ranges, prices like $45, $50)
  * "nearest_10" - round to nearest 10 (engine displacement, large measurements)
  * Leave null for precise continuous values that should not be rounded

DATASET NAME:
- Generate a descriptive but file-safe dataset_name (e.g., "car_fuel_efficiency", "customer_churn_prediction")
- Use lowercase and underscores only, no spaces or special characters
"""
    
    return prompt


def generate_fallback_plan(answers: WizardAnswers) -> DatasetPlan:
    """Generate a simple fallback plan if LLM fails."""
    
    logger.info("Using fallback plan generation")
    
    # Use 8 features as default fallback
    num_features = 8

    if answers.task == "classification":
        target_formula = "-1.0 + " + " + ".join([f"0.5*feature_{i+1}" for i in range(min(3, num_features))])
    else:
        target_formula = "100 + " + " + ".join([f"{5*(i+1)}*feature_{i+1}" for i in range(min(3, num_features))]) + " + normal(0,10)"
    
    # Generate a simple dataset name from description
    description = answers.custom_description or "generic dataset"
    dataset_name = description.lower().replace(" ", "_").replace("-", "_")
    # Remove special characters and limit length
    dataset_name = "".join(c for c in dataset_name if c.isalnum() or c == "_")[:50]
    if not dataset_name:
        dataset_name = "fallback_dataset"
    
    return DatasetPlan(
        task=answers.task,
        name="FallbackDataset",
        description=f"A simple {answers.task} dataset: {answers.custom_description or 'generic'}",
        dataset_name=dataset_name,
        features=[
            {
                "name": f"feature_{i+1}",
                "type": "numerical",
                "distribution": "normal(0, 1)",
                "domain_semantics": f"synthetic feature {i+1}",
                "missing_rate": 0.05 if i == 1 else 0.0,  # Add some missing values to feature_2
                "rounding_precision": "0.01"  # Default to 2 decimal places for fallback
            }
            for i in range(num_features)
        ],
        target_name="target",
        target_formula=target_formula,
        domain=answers.domain,
        seed=answers.seed
    )