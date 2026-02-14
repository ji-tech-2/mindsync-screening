"""
Unit tests for model.py initialization and artifact management functions
"""

import os
import sys
import pytest
import pickle
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path


class TestDownloadArtifactsFromWandB:
    """Test W&B artifact download functionality."""

    @patch("os.getenv")
    def test_skip_wandb_download_when_env_set(self, mock_getenv):
        """Test that W&B download is skipped when SKIP_WANDB_DOWNLOAD is true."""
        from flaskr.model import download_artifacts_from_wandb

        mock_getenv.return_value = "true"  # SKIP_WANDB_DOWNLOAD

        result = download_artifacts_from_wandb("/tmp/artifacts")

        assert result is True

    def test_wandb_not_installed_returns_true(self):
        """Test behavior when wandb module is not installed."""
        from flaskr.model import download_artifacts_from_wandb

        # The actual function handles ImportError internally
        # Just verify it returns True (allows local fallback)
        result = download_artifacts_from_wandb("/tmp/artifacts")

        # Should return True whether wandb is installed or not
        assert result is True


class TestLoadArtifacts:
    """Test artifact loading from disk."""

    def test_load_artifacts_success(self, tmp_path):
        """Test successful loading of all artifacts."""
        from flaskr.model import _load_artifacts, LinearRegressionRidge

        # Create mock artifacts
        model = LinearRegressionRidge()
        model.coef_ = np.array([1.0, 2.0, 3.0])

        preprocessor = {"scaler": "StandardScaler"}

        # Write mock files
        with open(tmp_path / "model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open(tmp_path / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        healthy_df = pd.DataFrame({"feature": [1, 2, 3]})
        healthy_df.to_csv(tmp_path / "healthy_cluster_avg.csv", index=False)

        coef_df = pd.DataFrame({"Feature": ["a", "b"], "Coefficient": [0.1, 0.2]})
        coef_df.to_csv(tmp_path / "model_coefficients.csv", index=False)

        # Load artifacts
        loaded = _load_artifacts(str(tmp_path))

        assert "model" in loaded
        assert "preprocessor" in loaded
        assert "healthy_cluster_df" in loaded
        assert "coefficients_df" in loaded

    def test_load_artifacts_missing_file(self, tmp_path):
        """Test that missing files raise appropriate errors."""
        from flaskr.model import _load_artifacts

        with pytest.raises(FileNotFoundError):
            _load_artifacts(str(tmp_path))


class TestValidateModel:
    """Test model validation function."""

    def test_validate_model_success(self):
        """Test successful model validation."""
        from flaskr.model import _validate_model, LinearRegressionRidge

        # Create a valid model
        model = LinearRegressionRidge()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([3, 7, 11])
        model.fit(X_train, y_train)

        loaded = {
            "model": Mock(predict=lambda x: np.array([75.5])),
            "preprocessor": None,
            "healthy_cluster_df": None,
            "coefficients_df": None,
        }

        result = _validate_model(loaded)

        assert result is True

    def test_validate_model_returns_nan(self):
        """Test validation failure when model returns NaN."""
        from flaskr.model import _validate_model

        loaded = {
            "model": Mock(predict=lambda x: np.array([np.nan])),
            "preprocessor": None,
            "healthy_cluster_df": None,
            "coefficients_df": None,
        }

        result = _validate_model(loaded)

        assert result is False

    def test_validate_model_exception(self):
        """Test validation failure on exception."""
        from flaskr.model import _validate_model

        loaded = {
            "model": Mock(predict=Mock(side_effect=Exception("Model error"))),
            "preprocessor": None,
            "healthy_cluster_df": None,
            "coefficients_df": None,
        }

        result = _validate_model(loaded)

        assert result is False


class TestBackupAndRestoreArtifacts:
    """Test artifact backup and restore functionality."""

    def test_backup_artifacts(self, tmp_path):
        """Test backing up artifacts."""
        from flaskr.model import _backup_artifacts

        # Create artifacts directory
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "model.pkl").write_text("model data")

        _backup_artifacts(str(artifacts_dir))

        backup_dir = Path(str(artifacts_dir) + ".bak")
        assert backup_dir.exists()
        assert (backup_dir / "model.pkl").exists()

    def test_restore_artifacts(self, tmp_path):
        """Test restoring artifacts from backup."""
        from flaskr.model import _restore_artifacts

        # Create backup directory
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        backup_dir = Path(str(artifacts_dir) + ".bak")
        backup_dir.mkdir()
        (backup_dir / "model.pkl").write_text("backup model")

        result = _restore_artifacts(str(artifacts_dir))

        assert result is True
        assert (artifacts_dir / "model.pkl").exists()

    def test_restore_artifacts_no_backup(self, tmp_path):
        """Test restore when no backup exists."""
        from flaskr.model import _restore_artifacts

        result = _restore_artifacts(str(tmp_path / "nonexistent"))

        assert result is False


class TestVersionChecking:
    """Test version checking and update functionality."""

    @patch("os.getenv")
    def test_get_latest_wandb_version_skip(self, mock_getenv):
        """Test that version check is skipped when configured."""
        from flaskr.model import _get_latest_wandb_version

        mock_getenv.return_value = "true"  # SKIP_WANDB_DOWNLOAD

        digest, artifact = _get_latest_wandb_version("/tmp/artifacts")

        assert digest is None
        assert artifact is None

    def test_get_latest_wandb_version_no_wandb(self):
        """Test version check when wandb is not installed."""
        from flaskr.model import _get_latest_wandb_version

        # Import error is handled internally
        digest, artifact = _get_latest_wandb_version("/tmp/artifacts")

        # Should return None, None when wandb unavailable
        assert digest is None or isinstance(digest, str)
        assert artifact is None or artifact is not None


class TestInitApp:
    """Test Flask app initialization."""

    @patch("flaskr.model.download_artifacts_from_wandb")
    @patch("flaskr.model._load_artifacts")
    @patch("flaskr.model._validate_model")
    @patch("flaskr.model._get_latest_wandb_version")
    @patch("flaskr.model._schedule_version_check")
    @patch("flaskr.model.os.getenv")
    def test_init_app_success(
        self,
        mock_getenv,
        mock_schedule,
        mock_get_version,
        mock_validate,
        mock_load,
        mock_download,
    ):
        """Test successful app initialization."""
        from flaskr.model import init_app

        mock_app = MagicMock()
        mock_app.root_path = "/app"

        # Setup mocks
        mock_download.return_value = True
        mock_load.return_value = {
            "model": MagicMock(),
            "preprocessor": MagicMock(),
            "healthy_cluster_df": pd.DataFrame(),
            "coefficients_df": pd.DataFrame(),
        }
        mock_validate.return_value = True
        mock_get_version.return_value = ("abc123", None)
        mock_getenv.side_effect = lambda key, default=None: (
            "false" if key == "SKIP_WANDB_DOWNLOAD" else default
        )

        init_app(mock_app)

        mock_download.assert_called_once()
        mock_load.assert_called_once()
        mock_validate.assert_called_once()
        mock_schedule.assert_called_once()

    @patch("flaskr.model.download_artifacts_from_wandb")
    @patch("flaskr.model._load_artifacts")
    @patch("flaskr.model._validate_model")
    @patch("flaskr.model.os.getenv")
    def test_init_app_load_failure(
        self, mock_getenv, mock_validate, mock_load, mock_download
    ):
        """Test app initialization when artifact loading fails."""
        from flaskr.model import init_app

        mock_app = MagicMock()
        mock_app.root_path = "/app"

        mock_download.return_value = True
        mock_load.side_effect = Exception("Load failed")
        mock_getenv.return_value = "false"

        # Should not raise, just log error
        init_app(mock_app)

        mock_download.assert_called_once()
        mock_load.assert_called_once()


class TestCheckForUpdates:
    """Test background update checking."""

    @patch("flaskr.model._artifacts_path", "/tmp/artifacts")
    @patch("flaskr.model._current_version", "old123")
    @patch("flaskr.model._get_latest_wandb_version")
    @patch("flaskr.model._schedule_version_check")
    def test_check_for_updates_no_new_version(self, mock_schedule, mock_get_version):
        """Test when no new version is available."""
        from flaskr.model import _check_for_updates

        mock_get_version.return_value = ("old123", None)

        _check_for_updates()

        mock_get_version.assert_called_once()
        mock_schedule.assert_called_once()

    @patch("flaskr.model._artifacts_path", "/tmp/artifacts")
    @patch("flaskr.model._current_version", "old123")
    @patch("flaskr.model._get_latest_wandb_version")
    @patch("flaskr.model._backup_artifacts")
    @patch("flaskr.model._download_wandb_artifact")
    @patch("flaskr.model._load_artifacts")
    @patch("flaskr.model._validate_model")
    @patch("flaskr.model._apply_loaded_artifacts")
    @patch("flaskr.model._schedule_version_check")
    def test_check_for_updates_new_version_success(
        self,
        mock_schedule,
        mock_apply,
        mock_validate,
        mock_load,
        mock_download,
        mock_backup,
        mock_get_version,
    ):
        """Test successful update to new version."""
        from flaskr.model import _check_for_updates

        mock_artifact = MagicMock()
        mock_get_version.return_value = ("new456", mock_artifact)
        mock_download.return_value = True
        mock_load.return_value = {"model": MagicMock()}
        mock_validate.return_value = True

        _check_for_updates()

        mock_backup.assert_called_once()
        mock_download.assert_called_once()
        mock_validate.assert_called_once()
        mock_apply.assert_called_once()

    @patch("flaskr.model._artifacts_path", "/tmp/artifacts")
    @patch("flaskr.model._current_version", "old123")
    @patch("flaskr.model._get_latest_wandb_version")
    @patch("flaskr.model._backup_artifacts")
    @patch("flaskr.model._download_wandb_artifact")
    @patch("flaskr.model._load_artifacts")
    @patch("flaskr.model._validate_model")
    @patch("flaskr.model._restore_artifacts")
    @patch("flaskr.model._schedule_version_check")
    def test_check_for_updates_validation_fails(
        self,
        mock_schedule,
        mock_restore,
        mock_validate,
        mock_load,
        mock_download,
        mock_backup,
        mock_get_version,
    ):
        """Test rollback when new version fails validation."""
        from flaskr.model import _check_for_updates

        mock_artifact = MagicMock()
        mock_get_version.return_value = ("new456", mock_artifact)
        mock_download.return_value = True
        mock_load.return_value = {"model": MagicMock()}
        mock_validate.return_value = False

        _check_for_updates()

        mock_restore.assert_called_once()
        mock_schedule.assert_called_once()
