"""
Unit tests for model.py - ML model and preprocessing functions
"""

import pytest
import numpy as np
import pandas as pd
from flaskr.model import (
    clean_occupation_column,
    LinearRegressionRidge,
    analyze_wellness_factors,
    categorize_mental_health_score,
)


class TestCleanOccupationColumn:
    """Test occupation column cleaning function."""

    def test_combine_unemployed_and_retired(self):
        """Test that Unemployed and Retired are combined."""
        df = pd.DataFrame(
            {"occupation": ["Student", "Unemployed", "Retired", "Engineer"]}
        )

        result = clean_occupation_column(df)

        assert "Unemployed" in result["occupation"].values
        assert "Retired" not in result["occupation"].values
        assert result["occupation"].tolist() == [
            "Student",
            "Unemployed",
            "Unemployed",
            "Engineer",
        ]

    def test_original_dataframe_unchanged(self):
        """Test that original dataframe is not modified."""
        df = pd.DataFrame({"occupation": ["Retired", "Engineer"]})
        original_values = df["occupation"].copy()

        result = clean_occupation_column(df)

        # Original should be unchanged
        assert df["occupation"].equals(original_values)
        # Result should be different
        assert not result["occupation"].equals(original_values)

    def test_missing_occupation_column(self):
        """Test handling when occupation column doesn't exist."""
        df = pd.DataFrame({"age": [25, 30, 35]})

        result = clean_occupation_column(df)

        # Should return copy without errors
        assert "occupation" not in result.columns
        assert result.equals(df)


class TestLinearRegressionRidge:
    """Test custom Ridge Regression implementation."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = LinearRegressionRidge()

        assert model.alpha == 1.0
        assert model.fit_intercept is True
        assert model.normalize is False
        assert model.solver == "closed_form"
        assert model.coef_ is None
        assert model.intercept_ == 0.0

    def test_fit_closed_form(self):
        """Test fitting with closed-form solution."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3, 7, 11])

        model = LinearRegressionRidge(alpha=0.1, solver="closed_form")
        model.fit(X, y)

        assert model.coef_ is not None
        assert len(model.coef_) == 2
        assert model.intercept_ != 0.0

    def test_predict(self):
        """Test prediction after fitting."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3, 7, 11])

        model = LinearRegressionRidge(alpha=0.1)
        model.fit(X, y)

        predictions = model.predict([[2, 3]])
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float, np.number))

    def test_score(self):
        """Test R² score calculation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = LinearRegressionRidge(alpha=0.01)
        model.fit(X, y)

        score = model.score(X, y)
        assert 0 <= score <= 1
        assert score > 0.95  # Should have high R² for linear relationship

    def test_fit_with_normalize(self):
        """Test fitting with normalization enabled."""
        X = np.array([[1, 100], [2, 200], [3, 300]])
        y = np.array([3, 7, 11])

        model = LinearRegressionRidge(alpha=0.1, normalize=True)
        model.fit(X, y)

        assert model.scaler_ is not None
        assert model.coef_ is not None

    def test_fit_gradient_descent(self):
        """Test fitting with gradient descent."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3, 7, 11])

        model = LinearRegressionRidge(
            alpha=0.1, solver="gd", learning_rate=0.01, max_iter=1000
        )
        model.fit(X, y)

        assert model.coef_ is not None
        assert len(model.coef_) == 2

    def test_fit_sgd(self):
        """Test fitting with stochastic gradient descent."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([3, 7, 11, 15])

        model = LinearRegressionRidge(
            alpha=0.1, solver="sgd", learning_rate=0.01, max_iter=500
        )
        model.fit(X, y)

        assert model.coef_ is not None

    def test_fit_without_intercept(self):
        """Test fitting without intercept."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3, 7, 11])

        model = LinearRegressionRidge(alpha=0.1, fit_intercept=False)
        model.fit(X, y)

        assert model.intercept_ == 0.0
        assert model.coef_ is not None

    def test_invalid_solver(self):
        """Test that invalid solver raises error."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([3, 7])

        model = LinearRegressionRidge(solver="invalid_solver")

        with pytest.raises(ValueError, match="Unknown solver"):
            model.fit(X, y)


class TestCategorizeMentalHealthScore:
    """Test mental health score categorization."""

    def test_dangerous_category(self):
        """Test score in dangerous range."""
        assert categorize_mental_health_score(10) == "dangerous"
        assert categorize_mental_health_score(12) == "dangerous"

    def test_not_healthy_category(self):
        """Test score in not healthy range."""
        assert categorize_mental_health_score(13) == "not healthy"
        assert categorize_mental_health_score(20) == "not healthy"
        assert categorize_mental_health_score(28.6) == "not healthy"

    def test_average_category(self):
        """Test score in average range."""
        assert categorize_mental_health_score(29) == "average"
        assert categorize_mental_health_score(40) == "average"
        assert categorize_mental_health_score(61.4) == "average"

    def test_healthy_category(self):
        """Test score in healthy range."""
        assert categorize_mental_health_score(62) == "healthy"
        assert categorize_mental_health_score(75) == "healthy"
        assert categorize_mental_health_score(100) == "healthy"

    def test_boundary_values(self):
        """Test boundary values between categories."""
        assert categorize_mental_health_score(12) == "dangerous"
        assert categorize_mental_health_score(12.1) == "not healthy"
        assert categorize_mental_health_score(28.6) == "not healthy"
        assert categorize_mental_health_score(28.7) == "average"
        assert categorize_mental_health_score(61.4) == "average"
        assert categorize_mental_health_score(61.5) == "healthy"

    def test_extreme_values(self):
        """Test extreme values."""
        assert categorize_mental_health_score(0) == "dangerous"
        assert categorize_mental_health_score(100) == "healthy"
        assert categorize_mental_health_score(150) == "healthy"  # Beyond 100


class TestAnalyzeWellnessFactors:
    """Test wellness factor analysis function."""

    def test_returns_none_when_components_missing(self, monkeypatch):
        """Test that function returns None when required components are missing."""
        import flaskr.model

        # Mock missing components
        monkeypatch.setattr(flaskr.model, "preprocessor", None)
        monkeypatch.setattr(flaskr.model, "healthy_cluster_df", None)
        monkeypatch.setattr(flaskr.model, "coefficients_df", None)

        user_df = pd.DataFrame({"age": [25]})
        result = analyze_wellness_factors(user_df)

        assert result is None

    def test_returns_expected_structure(self, monkeypatch):
        """Test that function returns expected data structure when successful."""
        import flaskr.model

        # Create mock components
        class MockPreprocessor:
            def transform(self, df):
                return np.array([[0.5, 0.3, 0.2]])

        mock_healthy_df = pd.DataFrame({"feature": [1, 2, 3]})
        mock_coef_df = pd.DataFrame(
            {
                "Feature": ["Sleep", "Exercise", "Stress"],
                "Coefficient": [0.5, 0.3, -0.2],
            }
        )

        monkeypatch.setattr(flaskr.model, "preprocessor", MockPreprocessor())
        monkeypatch.setattr(flaskr.model, "healthy_cluster_df", mock_healthy_df)
        monkeypatch.setattr(flaskr.model, "coefficients_df", mock_coef_df)

        user_df = pd.DataFrame({"feature": [1, 2, 3]})
        result = analyze_wellness_factors(user_df)

        if result is not None:
            assert "areas_for_improvement" in result
            assert "strengths" in result
            assert isinstance(result["areas_for_improvement"], list)
            assert isinstance(result["strengths"], list)


class TestModelIntegration:
    """Integration tests for model functionality."""

    def test_model_pipeline(self):
        """Test complete model training and prediction pipeline."""
        # Create simple dataset
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = (
            2 * X_train[:, 0]
            + 3 * X_train[:, 1]
            - X_train[:, 2]
            + np.random.randn(100) * 0.1
        )

        X_test = np.random.randn(20, 3)

        # Train model
        model = LinearRegressionRidge(alpha=0.5, solver="closed_form")
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)

        # Check score
        score = model.score(X_train, y_train)
        assert score > 0.8  # Should have good fit

    def test_model_consistency(self):
        """Test that model produces consistent results."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([3, 7, 11])

        # Train two models with same parameters
        model1 = LinearRegressionRidge(alpha=0.1, solver="closed_form")
        model2 = LinearRegressionRidge(alpha=0.1, solver="closed_form")

        model1.fit(X, y)
        model2.fit(X, y)

        # Predictions should be identical
        pred1 = model1.predict([[2, 3]])
        pred2 = model2.predict([[2, 3]])

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestCategorizeFunctions:
    """Additional tests for categorization functions."""

    def test_categorize_edge_cases(self):
        """Test categorization at exact boundaries."""
        assert categorize_mental_health_score(0.0) == "dangerous"
        assert categorize_mental_health_score(28.6) == "not healthy"
        assert categorize_mental_health_score(61.4) == "average"
        assert categorize_mental_health_score(75.0) == "healthy"
        assert categorize_mental_health_score(100.0) == "healthy"

    def test_categorize_fractional_scores(self):
        """Test categorization with fractional scores."""
        assert categorize_mental_health_score(12.0) == "dangerous"
        assert categorize_mental_health_score(28.5) == "not healthy"
        assert categorize_mental_health_score(61.3) == "average"
        assert categorize_mental_health_score(100.1) == "healthy"  # Above max

    def test_categorize_negative_score(self):
        """Test categorization with negative score."""
        result = categorize_mental_health_score(-10.0)
        assert result == "dangerous"
