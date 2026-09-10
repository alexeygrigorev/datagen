import logging
from openai import OpenAI
from .schemas import (
    BinaryFeature,
    CategoricalFeature,
    CategoricalTerm,
    ConstantTerm,
    DatasetPlan,
    LinearTerm,
    NormalDist,
    NumericalFeature,
    UniformDist,
    NoiseTerm,
    WizardAnswers,
    get_random_row_count,
)


logger = logging.getLogger(__name__)


def generate_dataset_plan(answers: WizardAnswers) -> DatasetPlan:
    """Generate dataset plan using OpenAI structured output."""

    prompt = build_prompt(answers)

    client = OpenAI()
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

        # Add the computed row count to the plan
        rows = get_random_row_count(answers.size, answers.seed)
        plan.rows = rows

        logger.info("Successfully generated dataset plan with structured output")
        return plan

    except Exception as e:
        logger.error(f"LLM structured output failed: {e}")
        return generate_fallback_plan(answers)


def build_prompt(answers: WizardAnswers) -> str:
    """Build prompt for LLM based on wizard answers.

    The DatasetPlan schema already constrains features, distributions, and
    target terms, so the prompt only needs high-level guidance.
    """

    rows = get_random_row_count(answers.size, answers.seed)
    description = answers.custom_description or "generic dataset"

    task_guidance = (
        "Design target_formula terms so the thresholded output (output >= 0 -> class 1) "
        "yields roughly balanced classes (30-70% positive)."
        if answers.task == "classification"
        else "Include a noise term in target_formula for realistic variability."
    )

    prompt = f"""Generate a synthetic dataset specification:

TASK: {answers.task}
DESCRIPTION: {description}
ROWS: {rows:,}
SEED: {answers.seed}

Create 5-20 realistic features with snake_case names, a target_name, and a file-safe dataset_name.
{task_guidance}
Give 2-4 features a missing_rate of 0.05-0.15, the rest 0.0.
Set rounding_precision from real-world context (integers for counts, 0.1 for ratings, 0.01 for prices, nearest_10/nearest_100 for large measurements, null to skip).
"""

    return prompt


def generate_fallback_plan(answers: WizardAnswers) -> DatasetPlan:
    """Generate a simple fallback plan if LLM fails."""

    logger.info("Using fallback plan generation")

    rows = get_random_row_count(answers.size, answers.seed)

    # Generate a simple dataset name from description
    description = answers.custom_description or "generic dataset"
    dataset_name = description.lower().replace(" ", "_").replace("-", "_")
    # Remove special characters and limit length
    dataset_name = "".join(c for c in dataset_name if c.isalnum() or c == "_")[:50]
    if not dataset_name:
        dataset_name = "fallback_dataset"

    features = [
        NumericalFeature(
            name=f"feature_{i+1}",
            distribution=NormalDist(mean=0, std=1),
            domain_semantics=f"synthetic feature {i+1}",
            missing_rate=0.05 if i == 1 else 0.0,
            rounding_precision="0.01",
        )
        for i in range(8)
    ]

    if answers.task == "classification":
        target_formula = [
            ConstantTerm(value=0.0),
            LinearTerm(feature="feature_1", coefficient=0.5),
            LinearTerm(feature="feature_2", coefficient=0.5),
            LinearTerm(feature="feature_3", coefficient=0.5),
        ]
    else:
        target_formula = [
            ConstantTerm(value=100.0),
            LinearTerm(feature="feature_1", coefficient=5.0),
            LinearTerm(feature="feature_2", coefficient=10.0),
            LinearTerm(feature="feature_3", coefficient=15.0),
            NoiseTerm(distribution=NormalDist(mean=0, std=10), coefficient=1.0),
        ]

    return DatasetPlan(
        task=answers.task,
        name="FallbackDataset",
        description=f"A simple {answers.task} dataset: {answers.custom_description or 'generic'}",
        dataset_name=dataset_name,
        features=features,
        target_name="target",
        target_formula=target_formula,
        domain=answers.domain,
        seed=answers.seed,
        rows=rows
    )
