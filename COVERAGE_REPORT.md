# Code Coverage Analysis Report

## Current Coverage Status

| Module | Coverage | Missing Lines | Status |
|--------|----------|---------------|---------|
| `datagen/__init__.py` | 100% | 0 | ✅ Complete |
| `datagen/formula_parsing.py` | 97% | 4 | ✅ Excellent |
| `datagen/schemas.py` | 90% | 4 | ✅ Good |
| `datagen/generator.py` | 49% | 151 | ⚠️ **Needs Coverage** |
| `datagen/main.py` | 12% | 106 | 🔴 **Critical Gap** |
| `datagen/llm_generator.py` | 0% | 37 | 🔴 **No Coverage** |
| `datagen/__main__.py` | 0% | 1 | ⚪ Entry point |

**Overall Coverage: 53% (303 of 638 lines missing)**

## Critical Areas Needing Test Coverage

### 🔥 Priority 1: Core Data Generation (`generator.py` - 49% coverage)

**Missing Feature Generation Methods:**
- Lines 88-114: Numerical distributions (normal, uniform, lognormal, poisson)
- Lines 115-163: Categorical, binary, ordinal, datetime generation
- Lines 297-351: Rounding precision logic (integer, nearest_5, nearest_10, etc.)
- Lines 353-421: Data quality features (missingness, outliers)

**Missing Target Generation:**
- Lines 189-256: Classification vs regression target creation
- Lines 422-485: Report generation and statistics

### 🔥 Priority 2: LLM Integration (`llm_generator.py` - 0% coverage)

**Completely Untested:**
- Lines 9-45: OpenAI API integration
- Lines 48-148: Prompt building and fallback plan generation

### 🔥 Priority 3: CLI Interface (`main.py` - 12% coverage)

**Missing User Workflows:**
- Lines 48-99: Interactive wizard and questionnaire
- Lines 119-265: File I/O, plan loading/saving, dataset export

## Recommended Testing Strategy

### Phase 1: Easy Wins (+30% coverage)

Create these test files:

1. **`tests/test_generator_features.py`**
   ```python
   # Test all feature generation methods
   # Test distribution parameter parsing
   # Test output shapes and value ranges
   ```

2. **`tests/test_data_quality.py`**
   ```python
   # Test rounding precision options
   # Test missingness injection rates
   # Test outlier generation
   ```

3. **Enhanced `tests/test_schemas.py`**
   ```python
   # Test get_random_row_count edge cases
   # Test SIZE_PRESETS validation
   ```

### Phase 2: Core Logic (+15% coverage)

4. **`tests/test_target_generation.py`**
   ```python
   # Test classification target thresholding
   # Test regression target with noise
   # Test formula evaluation fallbacks
   ```

5. **`tests/test_report_generation.py`**
   ```python
   # Test metadata collection
   # Test feature statistics computation
   ```

### Phase 3: External Dependencies (+10% coverage)

6. **`tests/test_llm_generator.py`**
   ```python
   # Mock OpenAI API calls
   # Test prompt building logic
   # Test fallback plan generation
   ```

7. **`tests/test_main_cli.py`**
   ```python
   # Mock questionary interactions
   # Test file operations with temp directories
   # Test end-to-end workflow
   ```

## Specific Missing Test Cases

### Feature Generation Edge Cases
- Empty distribution strings
- Invalid parameter counts
- Negative values for positive distributions
- Categorical features without categories specified

### Data Quality Edge Cases
- Zero missing rate vs 100% missing rate
- Rounding with very small/large numbers
- Outlier injection at different severity levels

### Error Handling
- Malformed formulas in target generation
- Missing features referenced in formulas
- File I/O errors in plan loading/saving

## Tools and Commands

**Run coverage analysis:**
```bash
uv run pytest --cov=datagen --cov-report=html --cov-report=term-missing tests/
```

**View detailed HTML report:**
```bash
# Open htmlcov/index.html in browser
```

**Track coverage improvements:**
```bash
uv run coverage report --show-missing
uv run coverage html
```

## Expected Outcomes

Following this testing strategy should bring coverage from **53% to ~90%** while focusing on the most critical, testable functionality:

- **Phase 1**: 53% → 83% (Core generation logic)
- **Phase 2**: 83% → 88% (Target and reporting)  
- **Phase 3**: 88% → 90% (External integrations)

## Next Steps

1. Start with Phase 1 tests (highest impact, easiest to implement)
2. Run coverage after each test file to track progress
3. Focus on business logic over CLI interactions initially
4. Use mocking for external dependencies (OpenAI API, file system)
5. Add integration tests once unit coverage is solid