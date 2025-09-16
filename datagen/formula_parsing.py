"""Formula parsing utilities for synthetic dataset generation."""

import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class ParsedTerm:
    """Base class for parsed formula terms."""
    coefficient: float


@dataclass 
class LinearTerm(ParsedTerm):
    """Linear term like '0.5*feature_name'."""
    feature_name: str


@dataclass
class CategoricalTerm(ParsedTerm):
    """Categorical indicator like '(feature==value)' or '[feature==value]'."""
    feature_name: str
    value: str


@dataclass
class ConstantTerm(ParsedTerm):
    """Constant term like '-2.5'."""
    pass


@dataclass
class NoiseTerm(ParsedTerm):
    """Noise term like 'normal(0,0.1)'."""
    distribution: str
    params: List[float]


class FormulaParser:
    """Parser for mathematical formulas used in synthetic data generation."""
    
    def __init__(self):
        # Patterns for different term types
        # Allow both explicit (*) and implicit multiplication
        self.linear_pattern = r'([+-]?\s*\d*\.?\d*)\s*\*?\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        self.categorical_pattern1 = r'([+-]?\s*\d*\.?\d*)\s*\*?\s*\[([^=]+)==([^\]]+)\]'
        self.categorical_pattern2 = r'([+-]?\s*\d*\.?\d*)\s*\*?\s*\(([^=]+)==[\'"]([^\'"]+)[\'"]\)'
        self.noise_pattern = r'([+-]?\s*\d*\.?\d*)\s*\*?\s*([a-zA-Z]+)\(([^)]+)\)'
        self.constant_pattern = r'([+-]?\s*\d+\.?\d*)(?!\s*[\*\[\(a-zA-Z])'
    
    def parse(self, formula: str) -> List[ParsedTerm]:
        """Parse a formula string into structured terms."""
        terms = []
        formula = formula.replace(' ', '')  # Remove all spaces for easier parsing
        
        # Parse in order of complexity to avoid conflicts
        # First parse categorical terms (most specific)
        cat_terms, formula = self._parse_and_remove_categorical_terms(formula)
        terms.extend(cat_terms)
        
        # Then parse noise terms
        noise_terms, formula = self._parse_and_remove_noise_terms(formula)
        terms.extend(noise_terms)
        
        # Then parse linear terms
        linear_terms, formula = self._parse_and_remove_linear_terms(formula)
        terms.extend(linear_terms)
        
        # Finally parse constants from remaining parts
        const_terms = self._parse_constant_terms(formula)
        terms.extend(const_terms)
        
        return terms
    
    def _parse_and_remove_categorical_terms(self, formula: str) -> tuple[List[CategoricalTerm], str]:
        """Parse categorical terms and remove them from the formula."""
        terms = []
        
        # Pattern 1: [feature==value]
        while True:
            match = re.search(self.categorical_pattern1, formula)
            if not match:
                break
            
            coef_str = match.group(1)
            feature_name = match.group(2).strip()
            value = match.group(3).strip()
            
            # Handle missing coefficient
            if not coef_str or coef_str in ['+', '-']:
                coef = 1.0 if coef_str != '-' else -1.0
            else:
                coef = float(coef_str)
            
            terms.append(CategoricalTerm(coefficient=coef, feature_name=feature_name, value=value))
            formula = formula[:match.start()] + formula[match.end():]
        
        # Pattern 2: (feature=='value')
        while True:
            match = re.search(self.categorical_pattern2, formula)
            if not match:
                break
            
            coef_str = match.group(1)
            feature_name = match.group(2).strip()
            value = match.group(3).strip()
            
            if not coef_str or coef_str in ['+', '-']:
                coef = 1.0 if coef_str != '-' else -1.0
            else:
                coef = float(coef_str)
            
            terms.append(CategoricalTerm(coefficient=coef, feature_name=feature_name, value=value))
            formula = formula[:match.start()] + formula[match.end():]
        
        return terms, formula
    
    def _parse_and_remove_noise_terms(self, formula: str) -> tuple[List[NoiseTerm], str]:
        """Parse noise terms and remove them from the formula."""
        terms = []
        
        while True:
            match = re.search(self.noise_pattern, formula)
            if not match:
                break
            
            coef_str = match.group(1)
            distribution = match.group(2)
            params_str = match.group(3)
            
            if not coef_str or coef_str in ['+', '-']:
                coef = 1.0 if coef_str != '-' else -1.0
            else:
                coef = float(coef_str)
            
            # Parse parameters
            params = [float(p.strip()) for p in params_str.split(',')]
            
            terms.append(NoiseTerm(coefficient=coef, distribution=distribution, params=params))
            formula = formula[:match.start()] + formula[match.end():]
        
        return terms, formula
    
    def _parse_and_remove_linear_terms(self, formula: str) -> tuple[List[LinearTerm], str]:
        """Parse linear terms and remove them from the formula."""
        terms = []
        
        while True:
            match = re.search(self.linear_pattern, formula)
            if not match:
                break
            
            coef_str = match.group(1)
            feature_name = match.group(2)
            
            # Handle missing coefficient (defaults to 1 or -1)
            if not coef_str or coef_str in ['+', '-']:
                coef = 1.0 if coef_str != '-' else -1.0
            else:
                coef = float(coef_str)
            
            terms.append(LinearTerm(coefficient=coef, feature_name=feature_name))
            formula = formula[:match.start()] + formula[match.end():]
        
        return terms, formula
    
    def _parse_constant_terms(self, formula: str) -> List[ConstantTerm]:
        """Parse constant terms from remaining formula string."""
        terms = []
        
        # Parse standalone numbers that weren't part of other terms
        matches = re.finditer(self.constant_pattern, formula)
        for match in matches:
            coef_str = match.group(1)
            coef = float(coef_str)
            terms.append(ConstantTerm(coefficient=coef))
        
        return terms


class FormulaEvaluator:
    """Evaluates parsed formulas against dataframes."""
    
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
    
    def evaluate(self, terms: List[ParsedTerm], df: pd.DataFrame) -> np.ndarray:
        """Evaluate parsed terms against a dataframe."""
        result = np.zeros(len(df))
        
        for term in terms:
            if isinstance(term, LinearTerm):
                if term.feature_name in df.columns:
                    result += term.coefficient * df[term.feature_name].values
            
            elif isinstance(term, CategoricalTerm):
                if term.feature_name in df.columns:
                    indicator = (df[term.feature_name] == term.value).astype(int)
                    result += term.coefficient * indicator
            
            elif isinstance(term, ConstantTerm):
                result += term.coefficient
            
            elif isinstance(term, NoiseTerm):
                noise = self._generate_noise(term, len(df))
                result += term.coefficient * noise
        
        return result
    
    def _generate_noise(self, term: NoiseTerm, size: int) -> np.ndarray:
        """Generate noise based on distribution specification."""
        if term.distribution == 'normal':
            if len(term.params) >= 2:
                return self.rng.normal(term.params[0], term.params[1], size)
            else:
                return self.rng.normal(0, 1, size)
        
        elif term.distribution == 'uniform':
            if len(term.params) >= 2:
                return self.rng.uniform(term.params[0], term.params[1], size)
            else:
                return self.rng.uniform(0, 1, size)
        
        else:
            # Fallback to normal(0,1)
            return self.rng.normal(0, 1, size)