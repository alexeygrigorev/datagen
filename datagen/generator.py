import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .schemas import (
    BinaryFeature,
    CategoricalFeature,
    CategoricalTerm,
    ConstantTerm,
    DatasetPlan,
    DerivedNumericalFeature,
    Feature,
    FeatureBounds,
    LinearTerm,
    MissingnessRule,
    LogNormalDist,
    NoiseTerm,
    NormalDist,
    NumericalDistribution,
    NumericalFeature,
    PoissonDist,
    TargetTerm,
    UniformDist,
    get_random_row_count,
)


logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Synthetic dataset generator based on LLM plan (structured terms only)."""

    def __init__(self, plan: DatasetPlan, rows: int, answers_dict: dict):
        self.plan = plan
        self.rows = rows
        self.answers = answers_dict
        self.rng = np.random.RandomState(plan.seed)

    def generate(self) -> Tuple[pd.DataFrame, Dict]:
        """Generate synthetic dataset and return DataFrame + report."""
        logger.info(f"Generating {self.rows} rows with {len(self.plan.features)} features")
        start_time = time.time()

        # Generate features in plan order. Derived features may depend on
        # previously generated columns, which makes domain relationships
        # explicit and deterministic.
        feature_data = {}
        for feature in self.plan.features:
            logger.info(f"Generating feature: {feature.name}")
            if isinstance(feature, DerivedNumericalFeature):
                feature_data[feature.name] = self._generate_derived_feature(
                    feature, pd.DataFrame(feature_data)
                )
            else:
                feature_data[feature.name] = self._generate_feature(feature)

        df = pd.DataFrame(feature_data)

        # Enforce physical/business bounds before rounding. This prevents
        # impossible intermediate values from influencing the target.
        df = self._apply_bounds(df)

        # Apply rounding to numerical features
        df = self._apply_rounding(df)

        # Generate target with correct column name
        if self.plan.task == "classification":
            target = self._generate_classification_target(df)
        else:
            target = self._generate_regression_target(df)

        df[self.plan.target_name] = target
        df = self._apply_target_rounding(df)

        # Apply missingness
        df = self._apply_missingness(df)

        # Apply outliers
        df = self._apply_outliers(df)

        # The legacy global outlier option is still supported, but a bounded
        # plan must never leave physically impossible values behind.
        df = self._apply_bounds(df)

        # Shuffle rows
        df = df.sample(frac=1, random_state=self.plan.seed).reset_index(drop=True)

        # Generate report
        generation_time = time.time() - start_time
        report = self._generate_report(df, generation_time)

        logger.info(f"Dataset generated in {generation_time:.2f}s")
        return df, report

    def _generate_feature(self, feature: Feature) -> np.ndarray:
        """Generate a single feature based on its structured spec."""
        if isinstance(feature, NumericalFeature):
            return self._generate_numerical_feature(feature.distribution)
        elif isinstance(feature, CategoricalFeature):
            return self._generate_categorical_feature(feature)
        elif isinstance(feature, BinaryFeature):
            return self._generate_binary_feature(feature.p)
        else:
            raise ValueError(f"Unsupported feature type: {getattr(feature, 'type', type(feature))}")

    def _generate_derived_feature(
        self, feature: DerivedNumericalFeature, df: pd.DataFrame
    ) -> np.ndarray:
        """Generate a numeric feature from a declared dependency formula."""
        referenced = self._referenced_features(feature.formula)
        missing = sorted(set(referenced) - set(df.columns))
        if missing:
            raise ValueError(
                f"Derived feature '{feature.name}' references features that have "
                f"not been generated yet: {', '.join(missing)}"
            )

        return self._evaluate_formula(feature.formula, df)

    @staticmethod
    def _referenced_features(terms: List[TargetTerm]) -> List[str]:
        """Return feature names referenced by structured formula terms."""
        references = []
        for term in terms:
            if isinstance(term, (LinearTerm, CategoricalTerm)):
                references.append(term.feature)
        return references

    def _generate_numerical_feature(self, dist: NumericalDistribution) -> np.ndarray:
        """Generate numerical feature from structured distribution."""
        if isinstance(dist, NormalDist):
            return self.rng.normal(dist.mean, dist.std, self.rows)
        elif isinstance(dist, UniformDist):
            return self.rng.uniform(dist.low, dist.high, self.rows)
        elif isinstance(dist, LogNormalDist):
            return self.rng.lognormal(dist.mean, dist.sigma, self.rows)
        elif isinstance(dist, PoissonDist):
            return self.rng.poisson(dist.lam, self.rows).astype(float)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

    def _generate_categorical_feature(self, feature: CategoricalFeature) -> np.ndarray:
        """Generate categorical feature."""
        if not feature.categories:
            return self.rng.choice(['A', 'B'], self.rows)

        labels = feature.categories
        probs = feature.probabilities
        if probs is None or len(probs) != len(labels):
            probs = [1.0 / len(labels)] * len(labels)

        # Normalize probabilities just in case
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()

        return self.rng.choice(labels, self.rows, p=probs)

    def _generate_binary_feature(self, p: float = 0.5) -> np.ndarray:
        """Generate binary feature."""
        return self.rng.binomial(1, p, self.rows)

    def _generate_classification_target(self, df: pd.DataFrame) -> np.ndarray:
        """Generate a classification target from a latent score.

        Existing plans keep the historical threshold behavior. New plans can
        request calibrated Bernoulli sampling from a logistic probability,
        which preserves uncertainty near the decision boundary and avoids a
        brittle hard threshold at zero.
        """
        formula_output = self._evaluate_formula(self.plan.target_formula, df)
        config = self.plan.classification

        if config and config.mode == "bernoulli_logistic":
            temperature = max(float(config.temperature), 1e-9)
            score = formula_output.astype(float)

            if config.target_rate is not None:
                score = score + self._calibration_shift(
                    score, float(config.target_rate), temperature
                )

            probabilities = 1.0 / (1.0 + np.exp(-score / temperature))
            probabilities = np.clip(
                probabilities,
                config.probability_clip_low,
                config.probability_clip_high,
            )
            targets = self.rng.binomial(1, probabilities).astype(int)
        else:
            # Backward-compatible behavior for legacy plans.
            targets = (formula_output >= 0).astype(int)

        logger.info(f"Generated {targets.sum()}/{len(targets)} positive class samples ({targets.mean():.2%})")
        return targets

    @staticmethod
    def _calibration_shift(
        score: np.ndarray, target_rate: float, temperature: float
    ) -> float:
        """Find an intercept shift whose mean logistic probability hits a target rate."""
        if not 0.0 < target_rate < 1.0:
            raise ValueError("classification.target_rate must be between 0 and 1")

        low, high = -50.0, 50.0
        for _ in range(80):
            midpoint = (low + high) / 2.0
            probabilities = 1.0 / (1.0 + np.exp(-(score + midpoint) / temperature))
            if probabilities.mean() < target_rate:
                low = midpoint
            else:
                high = midpoint
        return (low + high) / 2.0

    def _generate_regression_target(self, df: pd.DataFrame) -> np.ndarray:
        """Generate regression target using structured formula."""
        targets = self._evaluate_formula(self.plan.target_formula, df)

        # Add noise based on noise level
        noise_level = self.answers.get('noise_level', "medium")
        noise_scale = self._get_noise_scale(targets, noise_level)
        noise = self.rng.normal(0, noise_scale, len(targets))

        return targets + noise

    def _evaluate_formula(self, terms: List[TargetTerm], df: pd.DataFrame) -> np.ndarray:
        """Evaluate structured target terms against a dataframe. No string parsing."""
        result = np.zeros(len(df))

        for term in terms:
            if isinstance(term, LinearTerm):
                if term.feature in df.columns:
                    result += term.coefficient * df[term.feature].values

            elif isinstance(term, CategoricalTerm):
                if term.feature in df.columns:
                    indicator = (df[term.feature] == term.value).astype(int)
                    result += term.coefficient * indicator

            elif isinstance(term, ConstantTerm):
                result += term.value

            elif isinstance(term, NoiseTerm):
                noise = self._generate_noise(term, len(df))
                result += term.coefficient * noise

        return result

    def _generate_noise(self, term: NoiseTerm, size: int) -> np.ndarray:
        """Generate noise from structured distribution."""
        dist = term.distribution
        if isinstance(dist, NormalDist):
            return self.rng.normal(dist.mean, dist.std, size)
        elif isinstance(dist, UniformDist):
            return self.rng.uniform(dist.low, dist.high, size)
        else:
            return self.rng.normal(0, 1, size)

    def _get_noise_scale(self, targets: np.ndarray, noise_level: str) -> float:
        """Get noise scale relative to target standard deviation."""
        if self.plan.target_noise_scale is not None:
            return float(self.plan.target_noise_scale)

        target_std = np.std(targets)

        if noise_level == "low":
            return target_std * 0.05
        elif noise_level == "medium":
            return target_std * 0.15
        elif noise_level == "high":
            return target_std * 0.30
        else:
            return target_std * 0.15

    def _apply_rounding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply appropriate rounding to numerical features based on their precision specification."""
        df_result = df.copy()

        for feature in self.plan.features:
            if isinstance(feature, (NumericalFeature, DerivedNumericalFeature)) and feature.name in df_result.columns:
                rounding_precision = feature.rounding_precision

                if rounding_precision:
                    column_data = df_result[feature.name]

                    if rounding_precision == "integer":
                        # Use nullable integer type to handle missing values
                        df_result[feature.name] = np.round(column_data).astype('Int64')
                        logger.info(f"Rounded {feature.name} to integers")

                    elif rounding_precision == "1":
                        # Use nullable integer type to handle missing values
                        df_result[feature.name] = np.round(column_data, 0).astype('Int64')
                        logger.info(f"Rounded {feature.name} to whole numbers")

                    elif rounding_precision in ["0.1", "0.01", "0.001"]:
                        decimal_places = len(rounding_precision.split('.')[1])
                        df_result[feature.name] = np.round(column_data, decimal_places)
                        logger.info(f"Rounded {feature.name} to {decimal_places} decimal places")

                    elif rounding_precision == "nearest_5":
                        # Use nullable integer type to handle missing values
                        df_result[feature.name] = (np.round(column_data / 5) * 5).astype('Int64')
                        logger.info(f"Rounded {feature.name} to nearest 5")

                    elif rounding_precision == "nearest_10":
                        # Use nullable integer type to handle missing values
                        df_result[feature.name] = (np.round(column_data / 10) * 10).astype('Int64')
                        logger.info(f"Rounded {feature.name} to nearest 10")

                    elif rounding_precision == "nearest_25":
                        # Use nullable integer type to handle missing values
                        df_result[feature.name] = (np.round(column_data / 25) * 25).astype('Int64')
                        logger.info(f"Rounded {feature.name} to nearest 25")

                    elif rounding_precision == "nearest_50":
                        # Use nullable integer type to handle missing values
                        df_result[feature.name] = (np.round(column_data / 50) * 50).astype('Int64')
                        logger.info(f"Rounded {feature.name} to nearest 50")

                    elif rounding_precision == "nearest_100":
                        # Use nullable integer type to handle missing values
                        df_result[feature.name] = (np.round(column_data / 100) * 100).astype('Int64')
                        logger.info(f"Rounded {feature.name} to nearest 100")

                    else:
                        logger.warning(f"Unknown rounding precision '{rounding_precision}' for {feature.name}")

        return df_result

    def _apply_target_rounding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round a target only when the plan explicitly requests it."""
        precision = self.plan.target_rounding_precision
        if not precision or self.plan.target_name not in df.columns:
            return df

        result = df.copy()
        values = result[self.plan.target_name]
        if precision in ["0.1", "0.01", "0.001"]:
            decimals = len(precision.split(".")[1])
            result[self.plan.target_name] = np.round(values, decimals)
        elif precision in ["integer", "1"]:
            result[self.plan.target_name] = np.round(values).astype("Int64")
        else:
            logger.warning("Unknown target rounding precision '%s'", precision)
        return result

    def _apply_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip declared numeric features to their domain bounds."""
        result = df.copy()
        for feature in self.plan.features:
            if not isinstance(feature, (NumericalFeature, DerivedNumericalFeature)):
                continue
            if feature.name not in result.columns or feature.bounds is None:
                continue

            bounds: FeatureBounds = feature.bounds
            if bounds.low is not None:
                result[feature.name] = result[feature.name].clip(lower=bounds.low)
            if bounds.high is not None:
                result[feature.name] = result[feature.name].clip(upper=bounds.high)
        return result

    def _apply_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing values to specified features."""
        df_result = df.copy()

        # Apply per-feature missingness from plan
        for feature in self.plan.features:
            missing_rate = feature.missing_rate or 0.0
            if missing_rate > 0 and feature.name in df_result.columns:
                mask = self.rng.random(len(df_result)) < missing_rate
                df_result.loc[mask, feature.name] = np.nan
                logger.info(f"Applied {missing_rate:.1%} missing values to {feature.name}")

        # Apply conditional rules after the baseline rates. The condition is
        # evaluated on the fully generated data before this rule masks rows.
        for rule in self.plan.missingness_rules:
            if rule.feature not in df_result.columns:
                raise ValueError(
                    f"Missingness rule references unknown feature '{rule.feature}'"
                )
            if not 0.0 <= rule.rate <= 1.0:
                raise ValueError("Missingness rule rates must be between 0 and 1")

            condition = self._missingness_condition(df_result, rule)
            mask = condition & (self.rng.random(len(df_result)) < rule.rate)
            df_result.loc[mask, rule.feature] = np.nan
            logger.info(
                "Applied %.1f%% conditional missingness to %s",
                rule.rate * 100,
                rule.feature,
            )

        return df_result

    @staticmethod
    def _missingness_condition(df: pd.DataFrame, rule: MissingnessRule) -> np.ndarray:
        """Evaluate one optional row condition for a missingness rule."""
        if rule.condition_feature is None:
            return np.ones(len(df), dtype=bool)
        if rule.condition_feature not in df.columns:
            raise ValueError(
                f"Missingness condition references unknown feature '{rule.condition_feature}'"
            )

        series = df[rule.condition_feature]
        operator = rule.condition_operator or "eq"
        value = rule.condition_value
        if operator == "eq":
            return (series == value).to_numpy()
        if operator == "ne":
            return (series != value).to_numpy()
        if operator == "lt":
            return (series < value).fillna(False).to_numpy()
        if operator == "le":
            return (series <= value).fillna(False).to_numpy()
        if operator == "gt":
            return (series > value).fillna(False).to_numpy()
        if operator == "ge":
            return (series >= value).fillna(False).to_numpy()
        raise ValueError(f"Unsupported missingness operator '{operator}'")

    def _apply_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply outliers to numerical features."""
        df_result = df.copy()

        # Global outliers from wizard answers
        global_outliers = self.answers.get('outliers', "none")
        global_rate = self._get_outliers_rate(global_outliers)

        # Apply global outliers to numerical features
        if global_rate > 0:
            numerical_cols = df_result.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col != self.plan.target_name:
                    self._inject_outliers(df_result, col, global_rate)

        return df_result

    def _inject_outliers(self, df: pd.DataFrame, column: str, rate: float):
        """Inject outliers into a numerical column."""
        n_outliers = int(len(df) * rate)
        if n_outliers == 0:
            return

        outlier_indices = self.rng.choice(len(df), n_outliers, replace=False)

        # Generate outliers as extreme values (3+ standard deviations)
        col_mean = df[column].mean()
        col_std = df[column].std()

        outlier_values = self.rng.choice([-1, 1], n_outliers) * (
            3 + self.rng.exponential(1, n_outliers)
        ) * col_std + col_mean

        df.loc[outlier_indices, column] = outlier_values

    def _get_outliers_rate(self, level: str) -> float:
        """Convert outliers level to rate."""
        if level == "none":
            return 0.0
        elif level == "slight":
            return 0.02
        else:
            return 0.0

    def _generate_report(self, df: pd.DataFrame, generation_time: float) -> Dict:
        """Generate dataset report."""
        report = {
            "metadata": {
                "rows": len(df),
                "columns": len(df.columns),
                "generation_time_seconds": round(generation_time, 2),
                "seed": self.plan.seed,
                "domain": self.plan.domain,
                "task": self.plan.task,
                "plan_version": self.plan.plan_version,
            },
            "target_stats": {},
            "feature_stats": {},
            "relationship_stats": {},
            "data_quality": {
                "missingness_rates": {},
                "outlier_counts": {},
                "bound_violations": {},
            }
        }

        # Target statistics
        target_col = self.plan.target_name
        if self.plan.task == "classification":
            target_counts = df[target_col].value_counts().to_dict()
            report["target_stats"] = {
                "class_distribution": target_counts,
                "class_balance": df[target_col].mean()
            }
        else:
            report["target_stats"] = {
                "mean": float(df[target_col].mean()),
                "std": float(df[target_col].std()),
                "min": float(df[target_col].min()),
                "max": float(df[target_col].max())
            }

        # Feature statistics
        for col in df.columns:
            if col == self.plan.target_name:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                report["feature_stats"][col] = {
                    "type": "numerical",
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "missing_rate": float(df[col].isna().mean())
                }
            else:
                value_counts = df[col].value_counts().head(5).to_dict()
                report["feature_stats"][col] = {
                    "type": "categorical",
                    "unique_values": int(df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in value_counts.items()},
                    "missing_rate": float(df[col].isna().mean())
                }

        # Data quality metrics
        for col in df.columns:
            if col != self.plan.target_name:
                missing_rate = df[col].isna().mean()
                if missing_rate > 0:
                    report["data_quality"]["missingness_rates"][col] = float(missing_rate)

        for feature in self.plan.features:
            if isinstance(feature, DerivedNumericalFeature):
                for term in feature.formula:
                    if not isinstance(term, LinearTerm):
                        continue
                    if term.feature not in df.columns or feature.name not in df.columns:
                        continue
                    pair = df[[term.feature, feature.name]].dropna()
                    if len(pair) >= 2 and pd.api.types.is_numeric_dtype(pair[term.feature]):
                        report["relationship_stats"][f"{term.feature}->{feature.name}"] = {
                            "coefficient": term.coefficient,
                            "correlation": float(pair[term.feature].corr(pair[feature.name])),
                        }

            if not isinstance(feature, (NumericalFeature, DerivedNumericalFeature)):
                continue
            if feature.bounds is None or feature.name not in df.columns:
                continue

            bounds = feature.bounds
            values = df[feature.name]
            violations = np.zeros(len(values), dtype=bool)
            if bounds.low is not None:
                violations |= (values < bounds.low).fillna(False).to_numpy()
            if bounds.high is not None:
                violations |= (values > bounds.high).fillna(False).to_numpy()
            report["data_quality"]["bound_violations"][feature.name] = int(violations.sum())

        return report


def generate_dataset(plan_file: str, answers: dict, output_dir: str = ".") -> Tuple[str, str]:
    """Generate dataset from plan file."""

    # Load plan
    with open(plan_file, 'r') as f:
        plan_data = json.load(f)

    plan = DatasetPlan.model_validate(plan_data)

    # Use rows from plan if available, otherwise compute from size preset
    if plan.rows is not None:
        rows = plan.rows
    else:
        # Fallback for old plans without rows field
        size_preset = answers['size']
        rows = get_random_row_count(size_preset, answers['seed'])

    # Generate dataset
    generator = DatasetGenerator(plan, rows, answers)
    df, report = generator.generate()

    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Use custom dataset name from plan
    dataset_name = getattr(plan, 'dataset_name', 'dataset')
    dataset_file = output_path / f"{dataset_name}.csv"
    report_file = output_path / f"{dataset_name}_report.json"

    # Write files
    df.to_csv(dataset_file, index=False)

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Dataset saved to {dataset_file}")
    logger.info(f"Report saved to {report_file}")

    return str(dataset_file), str(report_file)
