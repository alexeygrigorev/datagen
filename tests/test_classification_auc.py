"""Test classification AUC performance with structured target terms."""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from datagen.generator import DatasetGenerator
from datagen.schemas import (
    CategoricalTerm,
    ConstantTerm,
    DatasetPlan,
    LinearTerm,
    NoiseTerm,
    NormalDist,
)


def evaluate_terms(terms, df, seed=42):
    plan = DatasetPlan(
        task="classification",
        name="AUC Test",
        description="test",
        dataset_name="auc_test",
        features=[],
        target_name="target",
        target_formula=terms,
        domain="generic",
        seed=seed,
        rows=len(df),
    )
    gen = DatasetGenerator(plan, len(df), {"task": "classification", "size": "small", "domain": "generic", "seed": seed})
    return gen._evaluate_formula(terms, df)


class TestClassificationAUC:
    """Test classification performance with structured formula."""

    def setup_method(self):
        np.random.seed(42)

    def test_formula_gives_good_auc(self):
        """Test that structured formula produces predictive targets (AUC > 0.65)."""
        terms = [
            ConstantTerm(value=-1.0),
            LinearTerm(feature='number_of_courses_viewed', coefficient=0.5),
            LinearTerm(feature='annual_income', coefficient=0.00001),
            LinearTerm(feature='interaction_count', coefficient=0.3),
            CategoricalTerm(feature='lead_source', value='paid_ads', coefficient=-1.0),
            CategoricalTerm(feature='lead_source', value='referral', coefficient=1.5),
            NoiseTerm(distribution=NormalDist(mean=0, std=0.1), coefficient=1.0),
        ]

        n_samples = 5000
        df = pd.DataFrame({
            'number_of_courses_viewed': np.random.poisson(2, n_samples),
            'annual_income': np.random.normal(60000, 15000, n_samples),
            'interaction_count': np.random.poisson(3, n_samples),
            'lead_source': np.random.choice(['social_media', 'referral', 'organic_search', 'events', 'paid_ads'], n_samples)
        })

        logits = evaluate_terms(terms, df)

        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        targets = np.random.binomial(1, probs, len(probs))

        assert targets.sum() > 0
        assert (targets == 0).sum() > 0
        assert 0.2 < targets.mean() < 0.8, f"Target balance should be reasonable, got {targets.mean():.3f}"

        X = df.copy()
        le = LabelEncoder()
        X['lead_source'] = le.fit_transform(X['lead_source'])

        X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.3, random_state=42)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        assert auc > 0.65, f"AUC should be > 0.65, got {auc:.3f}"

    def test_categorical_terms_impact_prediction(self):
        """Test that categorical terms have measurable impact on predictions."""
        terms = [
            CategoricalTerm(feature='lead_source', value='referral', coefficient=1.0),
            CategoricalTerm(feature='lead_source', value='paid_ads', coefficient=-1.0),
        ]

        df = pd.DataFrame({
            'lead_source': ['referral'] * 1000 + ['paid_ads'] * 1000 + ['other'] * 1000
        })

        result = evaluate_terms(terms, df)

        referral_scores = result[:1000]
        paid_ads_scores = result[1000:2000]
        other_scores = result[2000:]

        assert np.mean(referral_scores) > np.mean(other_scores)
        assert np.mean(other_scores) > np.mean(paid_ads_scores)
        assert np.mean(referral_scores) > np.mean(paid_ads_scores)

    def test_linear_terms_impact_prediction(self):
        """Test that linear terms have expected impact on predictions."""
        terms = [
            LinearTerm(feature='courses', coefficient=0.5),
            LinearTerm(feature='income', coefficient=0.1),
        ]

        df = pd.DataFrame({
            'courses': [1, 5, 10],
            'income': [30000, 60000, 90000]
        })

        result = evaluate_terms(terms, df)

        assert result[0] < result[1] < result[2], f"Predictions should increase: {result}"

        expected = [0.5*1 + 0.1*30000, 0.5*5 + 0.1*60000, 0.5*10 + 0.1*90000]
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_noise_adds_variability(self):
        """Test that noise terms add appropriate variability."""
        terms = [
            ConstantTerm(value=5.0),
            NoiseTerm(distribution=NormalDist(mean=0, std=1), coefficient=1.0),
        ]

        df = pd.DataFrame({'dummy': [1, 1, 1, 1, 1]})

        result = evaluate_terms(terms, df)

        assert 3.0 < np.mean(result) < 7.0, f"Mean should be around 5.0, got {np.mean(result):.3f}"
        assert np.std(result) > 0.5
        assert not np.allclose(result, result[0])
