"""Tests for formula parsing functionality."""


import numpy as np
import pandas as pd
from datagen.formula_parsing import (
    FormulaParser, FormulaEvaluator,
    LinearTerm, CategoricalTerm, ConstantTerm, NoiseTerm
)


class TestFormulaParser:
    """Test the FormulaParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = FormulaParser()
    
    def test_parse_linear_terms(self):
        """Test parsing of linear terms."""
        formula = "0.5*feature1 + 2*feature2 - 0.1*feature3"
        terms = self.parser.parse(formula)
        
        linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
        assert len(linear_terms) == 3
        
        # Check first term
        assert linear_terms[0].coefficient == 0.5
        assert linear_terms[0].feature_name == "feature1"
        
        # Check second term
        assert linear_terms[1].coefficient == 2.0
        assert linear_terms[1].feature_name == "feature2"
        
        # Check third term
        assert linear_terms[2].coefficient == -0.1
        assert linear_terms[2].feature_name == "feature3"
    
    def test_parse_categorical_terms_bracket_format(self):
        """Test parsing categorical terms with bracket format."""
        formula = "0.3*[category==value1] - 0.5*[category==value2]"
        terms = self.parser.parse(formula)
        
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        assert len(cat_terms) == 2
        
        assert cat_terms[0].coefficient == 0.3
        assert cat_terms[0].feature_name == "category"
        assert cat_terms[0].value == "value1"
        
        assert cat_terms[1].coefficient == -0.5
        assert cat_terms[1].feature_name == "category"
        assert cat_terms[1].value == "value2"
    
    def test_parse_categorical_terms_parentheses_format(self):
        """Test parsing categorical terms with parentheses format."""
        formula = "0.4*(lead_source=='referral') - 0.3*(lead_source=='paid_ads')"
        terms = self.parser.parse(formula)
        
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        assert len(cat_terms) == 2
        
        assert cat_terms[0].coefficient == 0.4
        assert cat_terms[0].feature_name == "lead_source"
        assert cat_terms[0].value == "referral"
        
        assert cat_terms[1].coefficient == -0.3
        assert cat_terms[1].feature_name == "lead_source"
        assert cat_terms[1].value == "paid_ads"
    
    def test_parse_constant_terms(self):
        """Test parsing of constant terms."""
        formula = "-2.5 + 1.0 - 0.7"
        terms = self.parser.parse(formula)
        
        const_terms = [t for t in terms if isinstance(t, ConstantTerm)]
        assert len(const_terms) == 3
        
        # Sort by coefficient for predictable testing
        const_terms.sort(key=lambda x: x.coefficient)
        
        assert const_terms[0].coefficient == -2.5
        assert const_terms[1].coefficient == -0.7
        assert const_terms[2].coefficient == 1.0
    
    def test_parse_noise_terms(self):
        """Test parsing of noise terms."""
        formula = "normal(0,0.1) + 0.5*uniform(0,1)"
        terms = self.parser.parse(formula)
        
        noise_terms = [t for t in terms if isinstance(t, NoiseTerm)]
        assert len(noise_terms) == 2
        
        # First term: normal(0,0.1)
        assert noise_terms[0].coefficient == 1.0
        assert noise_terms[0].distribution == "normal"
        assert noise_terms[0].params == [0, 0.1]
        
        # Second term: 0.5*uniform(0,1)
        assert noise_terms[1].coefficient == 0.5
        assert noise_terms[1].distribution == "uniform"
        assert noise_terms[1].params == [0, 1]
    
    def test_parse_complex_formula(self):
        """Test parsing a complex real-world formula."""
        formula = "-0.5 + 0.2*number_of_courses_viewed + 0.3*annual_income/100000 + 0.1*interaction_count - 0.3*(lead_source=='paid_ads') + 0.4*(lead_source=='referral') + normal(0,0.1)"
        terms = self.parser.parse(formula)
        
        # Check we got different types of terms
        linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        const_terms = [t for t in terms if isinstance(t, ConstantTerm)]
        noise_terms = [t for t in terms if isinstance(t, NoiseTerm)]
        
        # We should have:
        # - Linear: number_of_courses_viewed, interaction_count (annual_income/100000 won't parse as simple linear)
        # - Categorical: lead_source=='paid_ads', lead_source=='referral'
        # - Constant: -0.5
        # - Noise: normal(0,0.1)
        
        assert len(linear_terms) >= 2
        assert len(cat_terms) == 2
        assert len(const_terms) >= 1
        assert len(noise_terms) == 1
        
        # Check specific categorical terms
        paid_ads_term = next((t for t in cat_terms if t.value == "paid_ads"), None)
        assert paid_ads_term is not None
        assert paid_ads_term.coefficient == -0.3
        
        referral_term = next((t for t in cat_terms if t.value == "referral"), None)
        assert referral_term is not None
        assert referral_term.coefficient == 0.4
    
    def test_missing_coefficients(self):
        """Test handling of missing coefficients (should default to 1 or -1)."""
        formula = "feature1 - feature2 + [cat==val] - [cat==val2]"
        terms = self.parser.parse(formula)
        
        linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        
        # Check linear terms
        assert any(t.coefficient == 1.0 and t.feature_name == "feature1" for t in linear_terms)
        assert any(t.coefficient == -1.0 and t.feature_name == "feature2" for t in linear_terms)
        
        # Check categorical terms  
        assert any(t.coefficient == 1.0 and t.value == "val" for t in cat_terms)
        assert any(t.coefficient == -1.0 and t.value == "val2" for t in cat_terms)

    def test_realistic_mixed_formula_integration(self):
        """Test a realistic formula that combines different term types without explicit multiplication."""
        formula = "feature1 - feature2 + [cat==val] - [cat==val2]"
        terms = self.parser.parse(formula)
        
        # Should parse exactly 4 terms
        assert len(terms) == 4
        
        # Extract by type
        linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        
        # Verify counts
        assert len(linear_terms) == 2
        assert len(cat_terms) == 2
        
        # Verify specific linear terms with correct coefficients
        feature1_term = next((t for t in linear_terms if t.feature_name == "feature1"), None)
        feature2_term = next((t for t in linear_terms if t.feature_name == "feature2"), None)
        
        assert feature1_term is not None
        assert feature1_term.coefficient == 1.0
        assert feature2_term is not None
        assert feature2_term.coefficient == -1.0
        
        # Verify specific categorical terms with correct coefficients
        val_term = next((t for t in cat_terms if t.value == "val"), None)
        val2_term = next((t for t in cat_terms if t.value == "val2"), None)
        
        assert val_term is not None
        assert val_term.coefficient == 1.0
        assert val_term.feature_name == "cat"
        
        assert val2_term is not None
        assert val2_term.coefficient == -1.0
        assert val2_term.feature_name == "cat"

    def test_pattern_matching_robustness(self):
        """Test that regex patterns correctly identify their target structures."""
        import re
        
        # Test linear pattern matching on simple cases
        linear_examples = [
            ("2.5*feature", True),
            ("feature", True),  # implicit multiplication
            ("-feature", True),
            ("+0.1*var_name", True),
        ]
        
        for example, should_match in linear_examples:
            matches = list(re.finditer(self.parser.linear_pattern, example.replace(' ', '')))
            if should_match:
                assert len(matches) > 0, f"Linear pattern should match '{example}'"
        
        # Test categorical pattern matching  
        cat_examples = [
            ("[cat==val]", True),
            ("2*[feature==value]", True),
            ("-[status==active]", True),
        ]
        
        for example, should_match in cat_examples:
            matches = list(re.finditer(self.parser.categorical_pattern1, example.replace(' ', '')))
            if should_match:
                assert len(matches) > 0, f"Categorical pattern should match '{example}'"
        
        # Test that the full parser correctly handles overlapping patterns
        # by testing end-to-end parsing behavior rather than individual regex matches
        overlap_test = "[cat==val]"
        terms = self.parser.parse(overlap_test)
        
        # Should produce exactly one categorical term, no linear terms
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        linear_terms = [t for t in terms if isinstance(t, LinearTerm)]
        
        assert len(cat_terms) == 1, "Should parse as categorical term"
        assert len(linear_terms) == 0, "Should not create spurious linear terms"
        assert cat_terms[0].feature_name == "cat"
        assert cat_terms[0].value == "val"

    def test_parse_categorical_coefficient_edge_cases(self):
        """Test edge cases for categorical coefficient parsing."""
        # Test missing coefficient (should default to 1.0)
        formula = "[category==value]"
        terms = self.parser.parse(formula)
        
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        assert len(cat_terms) == 1
        assert cat_terms[0].coefficient == 1.0
        
        # Test negative coefficient without explicit value (should be -1.0)
        formula = "-[category==value]"
        terms = self.parser.parse(formula)
        
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        assert len(cat_terms) == 1
        assert cat_terms[0].coefficient == -1.0
        
        # Test positive coefficient without explicit value (should be 1.0)
        formula = "+[category==value]"
        terms = self.parser.parse(formula)
        
        cat_terms = [t for t in terms if isinstance(t, CategoricalTerm)]
        assert len(cat_terms) == 1
        assert cat_terms[0].coefficient == 1.0


class TestFormulaEvaluator:
    """Test the FormulaEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.RandomState(42)
        self.evaluator = FormulaEvaluator(self.rng)
        
        # Create test dataframe
        self.df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5],
            'category': ['A', 'B', 'A']
        })
    
    def test_evaluate_linear_terms(self):
        """Test evaluation of linear terms."""
        terms = [
            LinearTerm(coefficient=2.0, feature_name='feature1'),
            LinearTerm(coefficient=-0.5, feature_name='feature2')
        ]
        
        result = self.evaluator.evaluate(terms, self.df)
        
        # Expected: 2.0*[1,2,3] + (-0.5)*[0.5,1.5,2.5] = [2,4,6] + [-0.25,-0.75,-1.25] = [1.75,3.25,4.75]
        expected = np.array([1.75, 3.25, 4.75])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_categorical_terms(self):
        """Test evaluation of categorical terms."""
        terms = [
            CategoricalTerm(coefficient=1.0, feature_name='category', value='A'),
            CategoricalTerm(coefficient=-0.5, feature_name='category', value='B')
        ]
        
        result = self.evaluator.evaluate(terms, self.df)
        
        # Expected: 1.0*[1,0,1] + (-0.5)*[0,1,0] = [1,0,1] + [0,-0.5,0] = [1,-0.5,1]
        expected = np.array([1.0, -0.5, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_constant_terms(self):
        """Test evaluation of constant terms."""
        terms = [
            ConstantTerm(coefficient=2.5),
            ConstantTerm(coefficient=-1.0)
        ]
        
        result = self.evaluator.evaluate(terms, self.df)
        
        # Expected: 2.5 - 1.0 = 1.5 for all rows
        expected = np.array([1.5, 1.5, 1.5])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_noise_terms(self):
        """Test evaluation of noise terms."""
        # Set seed for reproducible noise
        self.evaluator.rng = np.random.RandomState(42)
        
        terms = [
            NoiseTerm(coefficient=1.0, distribution='normal', params=[0, 1]),
            NoiseTerm(coefficient=0.5, distribution='uniform', params=[0, 2])
        ]
        
        result = self.evaluator.evaluate(terms, self.df)
        
        # Should be 3 values (length of df), noise should vary
        assert len(result) == 3
        assert not np.allclose(result, result[0])  # Values should be different due to noise
    
    def test_evaluate_mixed_terms(self):
        """Test evaluation of mixed term types."""
        terms = [
            LinearTerm(coefficient=1.0, feature_name='feature1'),
            CategoricalTerm(coefficient=2.0, feature_name='category', value='A'),
            ConstantTerm(coefficient=0.5)
        ]
        
        result = self.evaluator.evaluate(terms, self.df)
        
        # Expected: feature1 + 2*(category=='A') + 0.5
        # Row 0: 1.0 + 2*1 + 0.5 = 3.5
        # Row 1: 2.0 + 2*0 + 0.5 = 2.5  
        # Row 2: 3.0 + 2*1 + 0.5 = 5.5
        expected = np.array([3.5, 2.5, 5.5])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_missing_features(self):
        """Test behavior when formula references missing features."""
        terms = [
            LinearTerm(coefficient=1.0, feature_name='missing_feature'),
            CategoricalTerm(coefficient=1.0, feature_name='missing_category', value='some_value')
        ]
        
        result = self.evaluator.evaluate(terms, self.df)
        
        # Should return zeros since features don't exist
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_noise_generation_edge_cases(self):
        """Test edge cases in noise term generation."""
        # Test normal distribution with insufficient parameters (should use default)
        term_insufficient = NoiseTerm(coefficient=1.0, distribution='normal', params=[5.0])  # Missing sigma
        result = self.evaluator._generate_noise(term_insufficient, 10)
        
        assert len(result) == 10
        # Should fallback to normal(0,1) since params are insufficient
        
        # Test uniform distribution with insufficient parameters
        term_uniform_insufficient = NoiseTerm(coefficient=1.0, distribution='uniform', params=[2.0])  # Missing upper bound
        result = self.evaluator._generate_noise(term_uniform_insufficient, 10)
        
        assert len(result) == 10
        # Should fallback to uniform(0,1)
        
        # Test unknown distribution (should fallback to normal)
        term_unknown = NoiseTerm(coefficient=1.0, distribution='unknown_dist', params=[1, 2])
        result = self.evaluator._generate_noise(term_unknown, 10)
        
        assert len(result) == 10
        # Should fallback to normal(0,1)


if __name__ == "__main__":
    pytest.main([__file__])
