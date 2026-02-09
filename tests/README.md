# Test Execution Guide

## Running Tests

### Run all tests:

```bash
pytest tests/ -v
```

### Run specific test file:

```bash
pytest tests/test_caching.py -v
pytest tests/test_model.py -v
pytest tests/test_cache.py -v
pytest tests/test_ai.py -v
```

### Run specific test class:

```bash
pytest tests/test_caching.py::TestWeeklyCriticalFactorsEndpoint -v
```

### Run specific test function:

```bash
pytest tests/test_model.py::TestLinearRegressionRidge::test_fit_closed_form -v
```

### Run tests with coverage (if pytest-cov installed):

```bash
pytest tests/ --cov=flaskr --cov-report=html
```

### Run only unit tests:

```bash
pytest tests/ -m unit
```

### Run tests in Docker:

```bash
docker build --target test -t mindsync-test .
```

## Test Structure

- `test_caching.py` - Tests for database caching (weekly factors, daily suggestions, chart data)
- `test_model.py` - Tests for ML model and preprocessing functions
- `test_cache.py` - Tests for Valkey/Redis caching functions
- `test_ai.py` - Tests for AI advice generation functions
- `test_predict_helpers.py` - Tests for predict.py helper functions

## Test Coverage

### Database Models (db.py)

- ✅ WeeklyCriticalFactors model
- ✅ WeeklyChartData model
- ✅ DailySuggestions model
- ✅ Model to_dict() methods
- ✅ UUID validation

### ML Model (model.py)

- ✅ clean_occupation_column function
- ✅ LinearRegressionRidge class (fit, predict, score)
- ✅ Multiple solvers (closed_form, gd, sgd)
- ✅ categorize_mental_health_score function
- ✅ analyze_wellness_factors function

### Cache (cache.py)

- ✅ store_prediction function
- ✅ fetch_prediction function
- ✅ update_prediction function
- ✅ is_available function
- ✅ Error handling

### AI (ai.py)

- ✅ get_ai_advice function
- ✅ get_weekly_advice function
- ✅ get_daily_advice function
- ✅ Error handling and fallbacks

### Endpoints (predict.py)

- ✅ /weekly-critical-factors with caching
- ✅ /chart/weekly with caching
- ✅ /daily-suggestion with caching
- ✅ format_db_output helper
- ✅ Cache hit/miss behavior

## Notes

- Tests use SQLite in-memory database for speed
- AI tests use mocked Gemini API calls
- Cache tests use mocked Valkey/Redis client
- All tests are isolated and can run independently
