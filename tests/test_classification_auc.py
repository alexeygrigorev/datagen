"""Test classification AUC performance with formula parser."""

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from datagen.formula_parsing import FormulaParser, FormulaEvaluator


class TestClassificationAUC:
    """Test classification performance with formula parsing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = FormulaParser()
        self.evaluator = FormulaEvaluator(np.random.RandomState(42))
        np.random.seed(42)  # For reproducible test data
    
    def test_formula_parsing_gives_good_auc(self):
        """Test that formula parsing produces predictive targets (AUC > 0.65)."""
        # Test formula with strong signal
        formula = "-1.0 + 0.5*number_of_courses_viewed + 0.00001*annual_income + 0.3*interaction_count - 1.0*(lead_source=='paid_ads') + 1.5*(lead_source=='referral') + normal(0,0.1)"
        
        # Create test data
        n_samples = 5000
        df = pd.DataFrame({
            'number_of_courses_viewed': np.random.poisson(2, n_samples),
            'annual_income': np.random.normal(60000, 15000, n_samples),
            'interaction_count': np.random.poisson(3, n_samples),
            'lead_source': np.random.choice(['social_media', 'referral', 'organic_search', 'events', 'paid_ads'], n_samples)
        })
        
        # Parse and evaluate formula
        terms = self.parser.parse(formula)
        logits = self.evaluator.evaluate(terms, df)
        
        # Convert to probabilities and binary targets
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        targets = np.random.binomial(1, probs, len(probs))
        
        # Should have balanced classes
        assert targets.sum() > 0, "Should have positive targets"
        assert (targets == 0).sum() > 0, "Should have negative targets"
        assert 0.2 < targets.mean() < 0.8, f"Target balance should be reasonable, got {targets.mean():.3f}"
        
        # Test with logistic regression
        X = df.copy()
        le = LabelEncoder()
        X['lead_source'] = le.fit_transform(X['lead_source'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.3, random_state=42)
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        
        y_pred_proba = lr.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Assert AUC is good (significantly better than random 0.5)
        assert auc > 0.65, f"AUC should be > 0.65 for good predictive power, got {auc:.3f}"
    
    def test_categorical_terms_impact_prediction(self):
        """Test that categorical terms have measurable impact on predictions."""
        formula = "1.0*(lead_source=='referral') - 1.0*(lead_source=='paid_ads')"
        
        # Create test data with specific categorical values
        df = pd.DataFrame({
            'lead_source': ['referral'] * 1000 + ['paid_ads'] * 1000 + ['other'] * 1000
        })
        
        terms = self.parser.parse(formula)
        result = self.evaluator.evaluate(terms, df)
        
        # Check that referral sources have higher scores
        referral_scores = result[:1000]
        paid_ads_scores = result[1000:2000]
        other_scores = result[2000:]
        
        assert np.mean(referral_scores) > np.mean(other_scores), "Referral should have higher scores than other"
        assert np.mean(other_scores) > np.mean(paid_ads_scores), "Other should have higher scores than paid_ads"
        assert np.mean(referral_scores) > np.mean(paid_ads_scores), "Referral should have much higher scores than paid_ads"
    
    def test_linear_terms_impact_prediction(self):
        """Test that linear terms have expected impact on predictions."""
        formula = "0.5*courses + 0.1*income"
        
        # Create test data with varying feature values
        df = pd.DataFrame({
            'courses': [1, 5, 10],
            'income': [30000, 60000, 90000]
        })
        
        terms = self.parser.parse(formula)
        result = self.evaluator.evaluate(terms, df)
        
        # Higher feature values should give higher predictions
        assert result[0] < result[1] < result[2], f"Predictions should increase with feature values: {result}"
        
        # Check approximate values
        expected = [0.5*1 + 0.1*30000, 0.5*5 + 0.1*60000, 0.5*10 + 0.1*90000]
        np.testing.assert_array_almost_equal(result, expected, decimal=1)
    
    def test_noise_adds_variability(self):
        """Test that noise terms add appropriate variability."""
        formula = "5.0 + normal(0,1)"
        
        # Create simple dataframe
        df = pd.DataFrame({'dummy': [1, 1, 1, 1, 1]})
        
        terms = self.parser.parse(formula)
        result = self.evaluator.evaluate(terms, df)
        
        # All values should be around 5.0 but with variation
        assert 3.0 < np.mean(result) < 7.0, f"Mean should be around 5.0, got {np.mean(result):.3f}"
        assert np.std(result) > 0.5, f"Should have noise variability, got std {np.std(result):.3f}"
        assert not np.allclose(result, result[0]), "Values should vary due to noise"


if __name__ == "__main__":
    pytest.main([__file__])