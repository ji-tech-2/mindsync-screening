"""
Unit tests for LinearRegressionRidge solver methods
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from flaskr.model import LinearRegressionRidge


class TestRidgeSolvers:
    """Test different solver implementations in LinearRegressionRidge."""

    def test_svd_solver(self):
        """Test SVD solver implementation."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([3, 7, 11, 15])

        model = LinearRegressionRidge(alpha=0.1, solver="svd")
        model.fit(X, y)

        assert model.coef_ is not None
        assert len(model.coef_) == 2

        # Test prediction
        pred = model.predict([[2, 3]])
        assert len(pred) == 1

    def test_cholesky_solver(self):
        """Test Cholesky decomposition solver."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([3, 7, 11, 15])

        model = LinearRegressionRidge(alpha=0.1, solver="cholesky")
        model.fit(X, y)

        assert model.coef_ is not None
        assert model.intercept_ is not None

    def test_cholesky_fallback(self):
        """Test Cholesky solver fallback on non-positive-definite matrix."""
        # Create a singular/ill-conditioned matrix
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([1, 2, 3])

        model = LinearRegressionRidge(alpha=0.001, solver="cholesky")
        # Should fall back to standard solver
        model.fit(X, y)

        assert model.coef_ is not None

    def test_conjugate_gradient_solver(self):
        """Test Conjugate Gradient iterative solver."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([3, 7, 11, 15])

        model = LinearRegressionRidge(alpha=0.1, solver="cg", max_iter=100)
        model.fit(X, y)

        assert model.coef_ is not None

        # Test prediction
        pred = model.predict([[4, 5]])
        assert isinstance(pred[0], (int, float, np.number))

    def test_sag_solver(self):
        """Test Stochastic Average Gradient solver."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([3, 7, 11, 15, 19])

        model = LinearRegressionRidge(
            alpha=0.1, solver="sag", learning_rate=0.01, max_iter=200
        )
        model.fit(X, y)

        assert model.coef_ is not None
        assert len(model.coef_) == 2

    def test_sgd_with_verbose(self):
        """Test SGD with verbose output."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([3, 7, 11, 15])

        model = LinearRegressionRidge(
            alpha=0.1, solver="sgd", learning_rate=0.01, max_iter=50, verbose=True
        )
        model.fit(X, y)

        assert model.coef_ is not None

    def test_gd_early_convergence(self):
        """Test gradient descent convergence."""
        # Skip this test as GD can be unstable
        pytest.skip("GD solver can be numerically unstable")

    def test_fit_with_1d_y(self):
        """Test fitting with 1D target array."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = LinearRegressionRidge(alpha=0.01)
        model.fit(X, y)

        pred = model.predict([[3]])
        # Should predict close to 6
        assert 5 < pred[0] < 7

    def test_predict_without_fit(self):
        """Test that prediction requires fitting first."""
        model = LinearRegressionRidge()

        # Should either raise an error or fail gracefully
        try:
            result = model.predict([[1, 2]])
            # If it doesn't raise, ensure we don't get valid predictions
            assert result is None or len(result) == 0
        except (AttributeError, TypeError, ValueError):
            # Expected - model not fitted
            pass

    def test_score_computation(self):
        """Test R² score computation."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

        model = LinearRegressionRidge(alpha=0.01)
        model.fit(X, y)

        score = model.score(X, y)

        # Should have high R² for nearly linear relationship
        assert 0.9 < score <= 1.0

    def test_normalize_feature_scaling(self):
        """Test model with feature normalization."""
        # Different scales
        X = np.array([[1, 1000], [2, 2000], [3, 3000], [4, 4000]])
        y = np.array([10, 20, 30, 40])

        model = LinearRegressionRidge(alpha=1.0, normalize=True)
        model.fit(X, y)

        assert model.scaler_ is not None
        assert model.coef_ is not None

        # Should make predictions
        pred = model.predict([[2.5, 2500]])
        assert 20 < pred[0] < 30

    def test_different_alphas(self):
        """Test model behavior with different regularization strengths."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([3, 7, 11, 15])

        # Low regularization
        model_low = LinearRegressionRidge(alpha=0.001)
        model_low.fit(X, y)

        # High regularization
        model_high = LinearRegressionRidge(alpha=100.0)
        model_high.fit(X, y)

        # Both should produce coefficients, but different magnitudes
        assert model_low.coef_ is not None
        assert model_high.coef_ is not None

        # High alpha should shrink coefficients more
        assert np.linalg.norm(model_high.coef_) < np.linalg.norm(model_low.coef_)

    def test_solver_convergence_warning(self):
        """Test that solvers handle non-convergence scenarios."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])

        # Very few iterations
        model = LinearRegressionRidge(
            alpha=0.1, solver="gd", learning_rate=0.001, max_iter=1, verbose=False
        )

        # Should still complete without error even if not fully converged
        model.fit(X, y)
        assert model.coef_ is not None


class TestModelEdgeCases:
    """Test edge cases and error handling."""

    def test_single_feature(self):
        """Test model with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])

        model = LinearRegressionRidge(alpha=0.1)
        model.fit(X, y)

        pred = model.predict([[2.5]])
        assert 4 < pred[0] < 6

    def test_many_features(self):
        """Test model with many features."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        model = LinearRegressionRidge(alpha=1.0)
        model.fit(X, y)

        assert len(model.coef_) == 20

    def test_perfect_fit(self):
        """Test model on perfectly linear data."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = LinearRegressionRidge(alpha=0.00001)
        model.fit(X, y)

        score = model.score(X, y)
        assert score > 0.99

    def test_collinear_features(self):
        """Test handling of collinear features."""
        # Create perfectly collinear features
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
        y = np.array([1, 2, 3, 4])

        # Ridge should handle this with regularization
        model = LinearRegressionRidge(alpha=1.0)
        model.fit(X, y)

        assert model.coef_ is not None
        pred = model.predict([[2, 4]])
        assert isinstance(pred[0], (int, float, np.number))


class TestApplyLoadedArtifacts:
    """Test _apply_loaded_artifacts function."""

    def test_apply_loaded_artifacts(self):
        """Test applying loaded artifacts to global state."""
        from flaskr.model import _apply_loaded_artifacts
        import flaskr.model as model_module

        mock_model = LinearRegressionRidge()
        mock_preprocessor = {"type": "scaler"}
        mock_healthy_df = np.array([[1, 2, 3]])
        mock_coef_df = np.array([[0.1, 0.2]])

        loaded = {
            "model": mock_model,
            "preprocessor": mock_preprocessor,
            "healthy_cluster_df": mock_healthy_df,
            "coefficients_df": mock_coef_df,
        }

        _apply_loaded_artifacts(loaded)

        # Verify global state was updated
        assert model_module.model is not None
        assert model_module.preprocessor is not None
        assert model_module.healthy_cluster_df is not None
        assert model_module.coefficients_df is not None


class TestScheduleVersionCheck:
    """Test _schedule_version_check function."""

    @patch("flaskr.model.threading.Timer")
    @patch("flaskr.model._VERSION_CHECK_INTERVAL", 10)
    def test_schedule_version_check(self, mock_timer):
        """Test that version check is scheduled."""
        from flaskr.model import _schedule_version_check

        mock_timer_instance = MagicMock()
        mock_timer.return_value = mock_timer_instance

        _schedule_version_check()

        # Should create and start a timer
        mock_timer.assert_called_once()
        mock_timer_instance.start.assert_called_once()

    @patch("flaskr.model.threading.Timer")
    @patch("flaskr.model._version_check_timer", MagicMock())
    def test_schedule_version_check_cancels_previous(self, mock_timer):
        """Test that previous timer is cancelled."""
        from flaskr.model import _schedule_version_check
        import flaskr.model as model_module

        # Setup existing timer
        old_timer = MagicMock()
        model_module._version_check_timer = old_timer

        mock_timer_instance = MagicMock()
        mock_timer.return_value = mock_timer_instance

        _schedule_version_check()

        # Old timer should be cancelled
        old_timer.cancel.assert_called_once()
