import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from schemas import (
    DatasetPlan, get_random_row_count
)


logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Synthetic dataset generator based on LLM plan."""
    
    def __init__(self, plan: DatasetPlan, rows: int, answers_dict: dict):
        self.plan = plan
        self.rows = rows
        self.answers = answers_dict
        self.rng = np.random.RandomState(plan.seed)
        
    def generate(self) -> Tuple[pd.DataFrame, Dict]:
        """Generate synthetic dataset and return DataFrame + report."""
        logger.info(f"Generating {self.rows} rows with {len(self.plan.features)} features")
        start_time = time.time()
        
        # Generate features
        feature_data = {}
        for feature in self.plan.features:
            logger.info(f"Generating feature: {feature.name}")
            feature_data[feature.name] = self._generate_feature(feature)
        
        df = pd.DataFrame(feature_data)
        
        # Apply rounding to numerical features
        df = self._apply_rounding(df)
        
        # Generate target with correct column name
        if self.plan.task == "classification":
            target = self._generate_classification_target(df)
        else:
            target = self._generate_regression_target(df)
        
        df[self.plan.target_name] = target
        
        # Apply missingness
        df = self._apply_missingness(df)
        
        # Apply outliers
        df = self._apply_outliers(df)
        
        # Shuffle rows
        df = df.sample(frac=1, random_state=self.plan.seed).reset_index(drop=True)
        
        # Generate report
        generation_time = time.time() - start_time
        report = self._generate_report(df, generation_time)
        
        logger.info(f"Dataset generated in {generation_time:.2f}s")
        return df, report
    
    def _generate_feature(self, feature) -> np.ndarray:
        """Generate a single feature based on its distribution."""
        dist_str = feature.distribution.strip()
        
        if feature.type == "numerical":
            return self._generate_numerical_feature(dist_str)
        elif feature.type == "categorical":
            return self._generate_categorical_feature(feature)
        elif feature.type == "binary":
            return self._generate_binary_feature(dist_str)
        elif feature.type == "ordinal":
            return self._generate_ordinal_feature(feature)
        elif feature.type == "datetime":
            return self._generate_datetime_feature(dist_str)
        else:
            raise ValueError(f"Unsupported feature type: {feature.type}")
    
    def _generate_numerical_feature(self, dist_str: str) -> np.ndarray:
        """Generate numerical feature from distribution string."""
        
        if dist_str.startswith('normal('):
            params = self._parse_params(dist_str, 'normal')
            mu, sigma = params[0], params[1]
            return self.rng.normal(mu, sigma, self.rows)
        
        elif dist_str.startswith('uniform('):
            params = self._parse_params(dist_str, 'uniform')
            a, b = params[0], params[1]
            return self.rng.uniform(a, b, self.rows)
        
        elif dist_str.startswith('lognormal('):
            params = self._parse_params(dist_str, 'lognormal')
            mu, sigma = params[0], params[1]
            return self.rng.lognormal(mu, sigma, self.rows)
        
        elif dist_str.startswith('poisson('):
            params = self._parse_params(dist_str, 'poisson')
            lam = params[0]
            return self.rng.poisson(lam, self.rows).astype(float)
        
        else:
            logger.warning(f"Unknown distribution {dist_str}, using normal(0,1)")
            return self.rng.normal(0, 1, self.rows)
    
    def _generate_categorical_feature(self, feature) -> np.ndarray:
        """Generate categorical feature."""
        if not feature.categories:
            # Fallback to binary if no categories specified
            return self.rng.choice(['A', 'B'], self.rows)
        
        # Handle both string lists and category objects
        if isinstance(feature.categories[0], str):
            # Simple string list - use equal probabilities
            labels = feature.categories
            probs = [1.0 / len(labels)] * len(labels)
        else:
            # Category objects with probabilities
            labels = [cat.label for cat in feature.categories]
            probs = [cat.p for cat in feature.categories]
        
        # Normalize probabilities just in case
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return self.rng.choice(labels, self.rows, p=probs)
    
    def _generate_binary_feature(self, dist_str: str) -> np.ndarray:
        """Generate binary feature."""
        if dist_str.startswith('bernoulli('):
            params = self._parse_params(dist_str, 'bernoulli')
            p = params[0]
        else:
            p = 0.5
        
        return self.rng.binomial(1, p, self.rows)
    
    def _generate_ordinal_feature(self, feature) -> np.ndarray:
        """Generate ordinal feature (treat as categorical for now)."""
        return self._generate_categorical_feature(feature)
    
    def _generate_datetime_feature(self, dist_str: str) -> np.ndarray:
        """Generate datetime feature."""
        # Simple implementation - random dates in 2023
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        
        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()
        
        timestamps = self.rng.uniform(start_timestamp, end_timestamp, self.rows)
        dates = pd.to_datetime(timestamps, unit='s')
        
        return dates.strftime('%Y-%m-%d')
    
    def _parse_params(self, dist_str: str, dist_name: str) -> List[float]:
        """Parse parameters from distribution string."""
        pattern = f"{dist_name}\\((.+?)\\)"
        match = re.search(pattern, dist_str)
        if not match:
            raise ValueError(f"Cannot parse distribution: {dist_str}")
        
        params_str = match.group(1)
        params = [float(x.strip()) for x in params_str.split(',')]
        return params
    
    def _apply_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply approximate correlations between numerical features."""
        # Simple implementation - find numerical columns and apply basic correlation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return df
        
        # Apply a simple correlation structure if specified
        # This is a simplified implementation
        logger.info("Applying correlations (simplified)")
        return df
    
    def _generate_classification_target(self, df: pd.DataFrame) -> np.ndarray:
        """Generate classification target using logit formula."""
        try:
            logit_formula = self.plan.target_formula
            logger.info(f"Using logit formula: {logit_formula}")
            
            # Simple evaluation of the logit formula
            logits = self._evaluate_formula(logit_formula, df)
            probs = expit(logits)  # sigmoid function
            
            # Apply class imbalance adjustment
            probs = self._adjust_class_balance(probs, "slight")
            
            # Generate binary targets
            targets = self.rng.binomial(1, probs, len(probs))
            return targets
            
        except Exception as e:
            logger.warning(f"Formula evaluation failed: {e}, using random targets")
            # Fallback to random with specified balance
            p = self._get_class_probability("slight")
            return self.rng.binomial(1, p, len(df))
    
    def _generate_regression_target(self, df: pd.DataFrame) -> np.ndarray:
        """Generate regression target using formula."""
        try:
            formula = self.plan.target_formula
            logger.info(f"Using regression formula: {formula}")
            
            targets = self._evaluate_formula(formula, df)
            
            # Add noise based on noise level
            noise_level = self.answers.get('noise_level', "medium")
            noise_scale = self._get_noise_scale(targets, noise_level)
            noise = self.rng.normal(0, noise_scale, len(targets))
            
            return targets + noise
            
        except Exception as e:
            logger.warning(f"Formula evaluation failed: {e}, using synthetic targets")
            # Fallback to simple linear combination
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                weights = self.rng.normal(0, 1, len(numerical_cols))
                targets = df[numerical_cols].values @ weights
                
                noise_level = self.answers.get('noise_level', "medium")
                noise_scale = self._get_noise_scale(targets, noise_level)
                noise = self.rng.normal(0, noise_scale, len(targets))
                
                return targets + noise
            else:
                return self.rng.normal(100, 20, len(df))
    
    def _evaluate_formula(self, formula: str, df: pd.DataFrame) -> np.ndarray:
        """Evaluate mathematical formula with feature values."""
        # Simple implementation - replace feature names with actual values
        # This is a basic version, could be much more sophisticated
        
        result = np.zeros(len(df))
        
        # Handle simple linear terms like "0.5*feature_name"
        for feature in self.plan.features:
            if feature.name in formula and feature.type == "numerical":
                # Extract coefficient for this feature
                pattern = r'([+-]?\s*\d*\.?\d*)\s*\*\s*' + re.escape(feature.name)
                matches = re.findall(pattern, formula)
                for match in matches:
                    coef = float(match.replace(' ', '') or '1')
                    result += coef * df[feature.name].values
        
        # Handle categorical indicators like "[feature==value]"
        categorical_pattern = r'\[([^=]+)==([^\]]+)\]'
        for match in re.finditer(categorical_pattern, formula):
            feature_name = match.group(1).strip()
            value = match.group(2).strip()
            
            if feature_name in df.columns:
                indicator = (df[feature_name] == value).astype(int)
                
                # Find coefficient before this indicator
                start_pos = max(0, match.start() - 10)
                before_text = formula[start_pos:match.start()]
                coef_match = re.search(r'([+-]?\s*\d*\.?\d*)$', before_text)
                coef = float(coef_match.group(1).replace(' ', '') or '1') if coef_match else 1.0
                
                result += coef * indicator
        
        # Handle constant terms
        constant_matches = re.findall(r'^([+-]?\d+\.?\d*)|(?<=[+-])\s*(\d+\.?\d*)(?!\s*\*)', formula)
        for match in constant_matches:
            constant = match[0] or match[1]
            if constant:
                result += float(constant)
        
        return result
    
    def _adjust_class_balance(self, probs: np.ndarray, class_prior: str) -> np.ndarray:
        """Adjust probabilities to achieve desired class balance."""
        target_p = self._get_class_probability(class_prior)
        
        # Simple adjustment - scale probabilities
        current_p = probs.mean()
        if current_p > 0:
            scale = target_p / current_p
            adjusted_probs = probs * scale
            adjusted_probs = np.clip(adjusted_probs, 0.01, 0.99)  # Keep in valid range
            return adjusted_probs
        
        return np.full_like(probs, target_p)
    
    def _get_class_probability(self, class_prior: str) -> float:
        """Get target probability for positive class."""
        if class_prior == "none":
            return 0.5
        elif class_prior == "slight":
            return 0.4
        elif class_prior == "moderate":
            return 0.3
        elif class_prior == "strong":
            return 0.2
        else:
            return 0.5
    
    def _get_noise_scale(self, targets: np.ndarray, noise_level: str) -> float:
        """Get noise scale relative to target standard deviation."""
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
            if feature.type == "numerical" and feature.name in df_result.columns:
                rounding_precision = getattr(feature, 'rounding_precision', None)
                
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
    
    def _apply_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing values to specified features."""
        df_result = df.copy()
        
        # Apply per-feature missingness from plan
        for feature in self.plan.features:
            missing_rate = getattr(feature, 'missing_rate', 0.0)
            if missing_rate > 0 and feature.name in df_result.columns:
                mask = self.rng.random(len(df_result)) < missing_rate
                df_result.loc[mask, feature.name] = np.nan
                logger.info(f"Applied {missing_rate:.1%} missing values to {feature.name}")
        
        return df_result
    
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
    
    def _get_missingness_rate(self, level: str) -> float:
        """Convert missingness level to rate."""
        if level == "none":
            return 0.0
        elif level == "slight":
            return 0.05
        elif level == "moderate":
            return 0.15
        else:
            return 0.0
    
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
                "task": self.plan.task
            },
            "target_stats": {},
            "feature_stats": {},
            "data_quality": {
                "missingness_rates": {},
                "outlier_counts": {}
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
                
            if df[col].dtype in [np.float64, np.int64]:
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
        
        return report


def generate_dataset(plan_file: str, answers: dict, output_dir: str = ".") -> Tuple[str, str]:
    """Generate dataset from plan file."""
    
    # Load plan
    with open(plan_file, 'r') as f:
        plan_data = json.load(f)
    
    plan = DatasetPlan.model_validate(plan_data)
    
    # Get randomized row count from size preset
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