import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import questionary
import typer
from dotenv import load_dotenv

from .schemas import WizardAnswers, get_random_row_count
from .providers import Provider, get_available_providers

load_dotenv()

app = typer.Typer(help="Synthetic ML Dataset Generator")
logger = logging.getLogger(__name__)


def setup_logging(outdir: str, level: str = "WARNING"):
    """Setup logging to file and console."""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    log_file = outdir_path / "syntheticgen.log"
    
    # File handler with INFO level for debugging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler with WARNING level to reduce noise
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def resolve_provider(cli_provider: Optional[str]) -> Optional[Provider]:
    """
    Determine which provider to use.
    Order of precedence:
    1. CLI argument (--provider)
    2. Interactive selection (if multiple keys found)
    3. Auto-detect (if only one key found)
    4. None (triggers fallback mode)
    """
    available = get_available_providers()
    
    # 1. CLI argument
    if cli_provider:
        try:
            return Provider(cli_provider.lower())
        except ValueError:
            typer.echo(f"Warning: Provider '{cli_provider}' not supported or configured. Available: {[p.value for p in available]}")
            # Fall through to auto-detect
            
    # If no keys found, return None (will use fallback)
    if not available:
        return None
        
    # 3. Auto-detect (single provider)
    if len(available) == 1:
        return available[0]
        
    # 2. Interactive selection (multiple providers)
    return questionary.select(
        "Select AI Provider:",
        choices=[p.value for p in available]
    ).ask()


def wizard_interview() -> WizardAnswers:
    """Interactive wizard to collect user requirements."""
    typer.echo("🎯 Synthetic ML Dataset Generator")
    typer.echo("Let's create your dataset specification...")
    
    # Size preset
    size = questionary.select(
        "Dataset size?",
        choices=[
            questionary.Choice("Small (~1K rows)", "small"),
            questionary.Choice("Medium (~10K rows)", "medium"),
            questionary.Choice("Large (~100K rows)", "large"),
            questionary.Choice("Very Large (~1M rows)", "very_large")
        ]
    ).ask()
    
    # Dataset description
    description = questionary.text(
        "Describe your dataset (e.g. 'customer churn prediction', 'car fuel efficiency', 'house prices'):",
        validate=lambda x: len(x.strip()) >= 5
    ).ask()
    
    # Auto-detect task type and determine other parameters
    description_lower = description.lower()
    if any(word in description_lower for word in ['churn', 'classification', 'predict whether', 'fraud', 'spam']):
        task = "classification"
    elif any(word in description_lower for word in ['price', 'cost', 'amount', 'efficiency', 'score', 'rating']):
        task = "regression"
    else:
        task = questionary.select(
            "Is this a classification or regression task?",
            choices=[
                questionary.Choice("Classification (predict categories)", "classification"),
                questionary.Choice("Regression (predict numbers)", "regression")
            ]
        ).ask()
    
    # Seed
    seed_input = questionary.text(
        "Random seed (press enter for auto):",
        default=""
    ).ask()
    seed = int(seed_input) if seed_input.strip() else random.randint(1, 1000000)
    
    return WizardAnswers(
        task=task,
        size=size,
        domain="generic",
        custom_description=description,
        seed=seed
    )


@app.command()
def main(
    size: Optional[str] = typer.Option(None, help="Size preset: small, medium, large, very_large"),
    description: Optional[str] = typer.Option(None, help="Dataset description"),
    task: Optional[str] = typer.Option(None, help="Task type: classification or regression"),
    seed: Optional[int] = typer.Option(None, help="Random seed"),
    plan: Optional[str] = typer.Option(None, help="Path to existing dataset plan JSON file"),
    provider: Optional[str] = typer.Option(None, help="AI Provider (openai, groq)"),
    accept: bool = typer.Option(False, help="Skip confirmation"),
    outdir: str = typer.Option("out", help="Output directory")
):
    """Generate synthetic ML datasets with LLM-assisted specification."""
    
    # Setup logging
    setup_logging(outdir)
    logger.info("Starting synthetic dataset generation")
    
    try:
        active_provider = resolve_provider(provider)
        
        # Handle existing plan file
        if plan:
            typer.echo("🎯 Loading existing dataset plan...")
            
            # Load plan from file
            with open(plan, 'r') as f:
                plan_data = json.load(f)
            
            from .schemas import DatasetPlan
            dataset_plan = DatasetPlan.model_validate(plan_data)
            
            # Create minimal answers for generation
            answers = WizardAnswers(
                task=dataset_plan.task,
                size="medium",  # Will be overridden by plan
                domain=dataset_plan.domain,
                custom_description=dataset_plan.description,
                seed=dataset_plan.seed,
                outdir=outdir,
                accept=accept
            )
            
            # Skip to generation step
            typer.echo(f"✅ Loaded plan: {dataset_plan.name}")
            typer.echo(f"   Features: {len(dataset_plan.features)}")
            typer.echo(f"   Target: {dataset_plan.target_name}")
            
        else:
            # Get requirements (wizard or CLI args)
            if not size or not description:
                answers = wizard_interview()
            else:
                # Build from CLI args
                answers = WizardAnswers(
                    task=task or "regression",
                    size=size,
                    domain="generic",
                    custom_description=description,
                    seed=seed or random.randint(1, 1000000),
                    outdir=outdir,
                    accept=accept
                )
            
            logger.info(f"Wizard answers: {answers}")
            
            # Use rows from plan if available, otherwise compute from size preset
            if plan and dataset_plan.rows is not None:
                rows = dataset_plan.rows
                typer.echo(f"\n📋 Dataset Summary (from saved plan):")
            else:
                # Get randomized row count from size preset
                rows = get_random_row_count(answers.size, answers.seed)
                typer.echo(f"\n📋 Dataset Summary:")
                
            typer.echo(f"   Task: {answers.task}")
            typer.echo(f"   Size: {answers.size} ({rows:,} rows)")
            typer.echo(f"   Description: {answers.custom_description}")
            typer.echo(f"   Seed: {answers.seed}")
            
            if not plan:
                # Generate LLM plan
                if active_provider:
                    typer.echo(f"\nGenerating dataset specification using {active_provider.value}...")
                    from .llm_generator import generate_dataset_plan
                    dataset_plan = generate_dataset_plan(answers, provider=active_provider)
                else:
                    typer.echo("\nNo AI keys found. Using deterministic fallback generator.")
                    from .llm_generator import generate_fallback_plan
                    dataset_plan = generate_fallback_plan(answers)
        
        # Show plan to user (more readable)
        if not plan:  # Only show detailed plan for new generations
            typer.echo(f"\n📋 Generated Plan:")
            typer.echo(f"   Name: {dataset_plan.name}")
            typer.echo(f"   Description: {dataset_plan.description}")
            typer.echo(f"   Features: {len(dataset_plan.features)}")
            
            # Show all features for approval
            typer.echo(f"   All features:")
            for feature in dataset_plan.features:
                feature_type_emoji = {"numerical": "🔢", "categorical": "📝", "binary": "⚡"}.get(feature.type, "📊")
                missing_info = f" (missing: {feature.missing_rate:.1%})" if getattr(feature, 'missing_rate', 0) > 0 else ""
                typer.echo(f"     {feature_type_emoji} {feature.name}: {feature.domain_semantics}{missing_info}")
            
            typer.echo(f"   Target: {dataset_plan.target_name}")
            typer.echo(f"   Target formula: {dataset_plan.target_formula}")
            
            # Ask for dataset name confirmation
            if not accept:
                proposed_name = getattr(dataset_plan, 'dataset_name', dataset_plan.name.lower().replace(' ', '_'))
                dataset_name = questionary.text(
                    f"Dataset filename (without .csv extension):",
                    default=proposed_name,
                    validate=lambda x: len(x.strip()) > 0 and all(c.isalnum() or c in '_-' for c in x.strip())
                ).ask()
                
                # Update the plan with confirmed name
                dataset_plan.dataset_name = dataset_name
            
            # Confirm or regenerate
            if not accept:
                choices = ["Accept and generate dataset"]
                if active_provider:
                    choices.append("Regenerate plan")
                choices.append("Cancel")

                action = questionary.select(
                    "What would you like to do?",
                    choices=choices
                ).ask()
                
                if action == "Cancel":
                    typer.echo("Cancelled.")
                    raise typer.Exit(0)
                elif action == "Regenerate plan":
                    typer.echo("🔄 Regenerating...")
                    from .llm_generator import generate_dataset_plan
                    dataset_plan = generate_dataset_plan(answers, provider=active_provider)
        
        # Save plan (only for new plans, not existing ones)
        if not plan:
            outdir_path = Path(outdir)
            outdir_path.mkdir(exist_ok=True)
            
            # Use dataset name for plan file
            plan_filename = f"{getattr(dataset_plan, 'dataset_name', 'dataset')}_plan.json"
            plan_file = outdir_path / plan_filename
            with open(plan_file, 'w') as f:
                json.dump(dataset_plan.model_dump(), f, indent=2)
            
            typer.echo(f"\n✅ Plan saved to {plan_file}")
        else:
            plan_file = plan
        
        # Generate dataset
        typer.echo("🎯 Generating synthetic dataset...")
        from .generator import generate_dataset
        
        answers_dict = answers.model_dump()
        dataset_file, report_file = generate_dataset(str(plan_file), answers_dict, outdir)
        
        # Show preview
        df = pd.read_csv(dataset_file)
        typer.echo(f"\n📊 Dataset Preview:")
        typer.echo(f"   Shape: {df.shape}")
        typer.echo(f"   Files: {dataset_file}, {report_file}")
        typer.echo(f"\n{df.head()}")
        
        logger.info("Generation completed successfully")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
