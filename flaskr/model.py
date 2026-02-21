"""
Machine Learning Model and Preprocessing
Custom Ridge Regression Implementation
"""

import os
import sys
import shutil
import pickle  # nosec B403 - Pickle used for trusted ML model files only
import logging
import threading
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Global variables for model components
model = None
preprocessor = None
healthy_cluster_df = None
coefficients_df = None

# Cluster DataFrames keyed by label
_cluster_dfs = {}  # e.g. {"dangerous": df, "not healthy": df, ...}

# Dynamic category thresholds (computed from cluster averages)
_category_thresholds = None  # List of (upper_bound, label) tuples, ascending

# Version tracking
_current_version = None  # W&B artifact digest of the loaded version
_artifacts_path = None  # Path to artifacts directory
_flask_app = None  # Flask app reference for background thread
_version_check_timer = None  # Background timer reference
_VERSION_CHECK_INTERVAL = 300  # Check every 5 minutes (seconds)


def download_artifacts_from_wandb(artifacts_path):
    """
    Attempt to download artifacts from Weights & Biases.

    Returns True if successful or if artifacts already exist locally.
    """
    logger.info("Starting W&B artifact download process")
    try:
        import wandb

        # Check if we should skip W&B download
        skip_wandb = os.getenv("SKIP_WANDB_DOWNLOAD", "false").lower() == "true"
        if skip_wandb:
            logger.info("Skipping W&B download (SKIP_WANDB_DOWNLOAD=true)")
            return True

        # Configuration from environment
        wandb_project = os.getenv("WANDB_PROJECT", "mindsync-model")
        wandb_entity = os.getenv("WANDB_ENTITY", None)
        artifact_name = "mindsync-model-smart"  # Match training artifact name
        artifact_version = os.getenv("ARTIFACT_VERSION", "latest")

        logger.info(
            f"W&B Configuration: project={wandb_project},"
            f"entity={wandb_entity}, version={artifact_version}"
        )

        # Initialize W&B API
        api = wandb.Api()

        # Construct artifact path
        if wandb_entity:
            artifact_path = (
                f"{wandb_entity}/{wandb_project}/{artifact_name}:{artifact_version}"
            )
        else:
            artifact_path = f"{wandb_project}/{artifact_name}:{artifact_version}"

        logger.info("Fetching W&B artifact: %s", artifact_path)

        # Check if healthy_cluster_avg.csv exists locally (should be preserved)
        healthy_cluster_path = os.path.join(artifacts_path, "healthy_cluster_avg.csv")
        backup_path = None
        if os.path.exists(healthy_cluster_path):
            backup_path = healthy_cluster_path + ".backup"
            import shutil

            shutil.copy2(healthy_cluster_path, backup_path)
            logger.info("Backed up local healthy_cluster_avg.csv")

        # Download artifact
        artifact = api.artifact(artifact_path, type="model")
        artifact_dir = artifact.download(root=artifacts_path)
        logger.info("Artifacts downloaded to: %s", artifact_dir)

        # Copy files from versioned folder to root artifacts/ for easy access
        from pathlib import Path

        versioned_path = Path(artifact_dir)
        artifacts_root = Path(artifacts_path)

        # Files to preserve (don't overwrite if they exist locally)
        preserve_files = ["healthy_cluster_avg.csv"]

        if versioned_path != artifacts_root:
            logger.info("Copying files from %s/ to artifacts/...", versioned_path.name)
            for file in versioned_path.glob("*.pkl"):
                dest = artifacts_root / file.name
                shutil.copy2(file, dest)
                logger.debug("Copied %s", file.name)
            for file in versioned_path.glob("*.csv"):
                if file.name not in preserve_files:
                    dest = artifacts_root / file.name
                    shutil.copy2(file, dest)
                    logger.debug("Copied %s", file.name)

            # Clean up versioned folder
            try:
                shutil.rmtree(versioned_path)
                logger.info("Cleaned up versioned folder: %s/", versioned_path.name)
            except Exception as e:
                logger.warning("Could not clean up %s: %s", versioned_path, e)

        # Restore local healthy_cluster_avg.csv if it was backed up
        if backup_path and os.path.exists(backup_path):
            shutil.move(backup_path, healthy_cluster_path)
            logger.info("Restored local healthy_cluster_avg.csv (not overwritten)")

        logger.info("W&B artifacts successfully downloaded to: %s", artifact_dir)
        return True

    except ImportError:
        logger.warning("wandb module not installed, skipping W&B download")
        return True

    except Exception as e:
        logger.error("Failed to download from W&B: %s", e)
        logger.info("Will attempt to use local artifacts if available")
        return True


# ===================== #
#  ARTIFACT LOADING     #
# ===================== #


def _load_artifacts(artifacts_path):
    """
    Load all model artifacts from disk into memory.
    Returns (model, preprocessor, healthy_cluster_df, coefficients_df) or raises.
    """
    # Register custom objects for pickle compatibility
    sys.modules["__main__"].clean_occupation_column = clean_occupation_column
    sys.modules["__main__"].LinearRegressionRidge = LinearRegressionRidge

    loaded = {}

    model_path = os.path.join(artifacts_path, "model.pkl")
    with open(model_path, "rb") as f:
        loaded["model"] = pickle.load(f)  # nosec B301 - trusted model file
    logger.info("Model loaded: %s", type(loaded["model"]).__name__)

    preprocessor_path = os.path.join(artifacts_path, "preprocessor.pkl")
    with open(preprocessor_path, "rb") as f:
        loaded["preprocessor"] = pickle.load(f)  # nosec B301 - trusted model file
    logger.info("Preprocessor loaded")

    healthy_path = os.path.join(artifacts_path, "healthy_cluster_avg.csv")
    loaded["healthy_cluster_df"] = pd.read_csv(healthy_path)
    logger.info("Healthy cluster data loaded")

    coef_path = os.path.join(artifacts_path, "model_coefficients.csv")
    loaded["coefficients_df"] = pd.read_csv(coef_path)
    logger.info("Coefficients loaded")

    # Load individual cluster average CSVs
    cluster_files = {
        "dangerous": "dangerous_cluster_avg.csv",
        "not healthy": "not_healthy_cluster_avg.csv",
        "average": "average_cluster_avg.csv",
        "above average": "above_average_cluster_avg.csv",
        "healthy": "healthy_cluster_avg.csv",
    }
    cluster_dfs = {}
    for label, filename in cluster_files.items():
        path = os.path.join(artifacts_path, filename)
        if os.path.exists(path):
            cluster_dfs[label] = pd.read_csv(path)
            logger.info("Cluster '%s' loaded from %s", label, filename)
        else:
            logger.warning("Cluster file not found: %s", path)
    loaded["cluster_dfs"] = cluster_dfs

    return loaded


def _validate_model(loaded):
    """
    Quick sanity check: ensure model can predict on a dummy input.
    Returns True if the model works, False otherwise.
    """
    try:
        test_model = loaded["model"]
        # Build a single-row DataFrame with the columns the pipeline expects
        test_data = pd.DataFrame(
            [
                {
                    "age": 25,
                    "gender": "Male",
                    "occupation": "Student",
                    "sleep_hours": 7.0,
                    "sleep_quality_1_5": 3,
                    "exercise_minutes_per_week": 150,
                    "social_hours_per_week": 10,
                    "stress_level_0_10": 5,
                    "work_screen_hours": 4,
                    "leisure_screen_hours": 3,
                    "productivity_0_100": 60,
                    "work_mode": "Hybrid",
                }
            ]
        )
        prediction = test_model.predict(test_data)
        if prediction is None or len(prediction) == 0:
            return False
        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
            return False
        logger.info("Model validation passed (test prediction: %.2f)", prediction[0])
        return True
    except Exception as e:
        logger.error("Model validation failed: %s", e)
        return False


def _apply_loaded_artifacts(loaded):
    """Apply loaded artifacts to global state."""
    global model, preprocessor, healthy_cluster_df, coefficients_df, _cluster_dfs
    model = loaded["model"]
    preprocessor = loaded["preprocessor"]
    healthy_cluster_df = loaded["healthy_cluster_df"]
    coefficients_df = loaded["coefficients_df"]
    _cluster_dfs = loaded["cluster_dfs"]


def _compute_category_thresholds():
    """
    Compute dynamic category thresholds from individual cluster CSVs.
    Reads mental_wellness_index_0_100 from each cluster CSV,
    then uses midpoints between adjacent scores as category boundaries.
    """
    global _category_thresholds

    if not _cluster_dfs:
        logger.warning("Cannot compute thresholds: cluster CSVs not loaded")
        return

    try:
        # Ordered category labels from lowest to highest wellness
        ordered_labels = [
            "dangerous",
            "not healthy",
            "average",
            "above average",
            "healthy",
        ]

        # Read the wellness index from each cluster CSV
        cluster_scores = {}
        for label in ordered_labels:
            if label in _cluster_dfs:
                df = _cluster_dfs[label]
                if "mental_wellness_index_0_100" in df.columns:
                    cluster_scores[label] = float(
                        df["mental_wellness_index_0_100"].iloc[0]
                    )
                else:
                    logger.warning(
                        "Cluster '%s' CSV missing mental_wellness_index_0_100", label
                    )

        # Sort by the predefined order
        sorted_scores = [
            (label, cluster_scores[label])
            for label in ordered_labels
            if label in cluster_scores
        ]

        # Compute midpoints between adjacent cluster scores as thresholds
        thresholds = []
        for i in range(len(sorted_scores) - 1):
            label = sorted_scores[i][0]
            current_score = sorted_scores[i][1]
            next_score = sorted_scores[i + 1][1]
            midpoint = (current_score + next_score) / 2.0
            thresholds.append((midpoint, label))

        # The last category has no upper bound — it catches everything above
        if sorted_scores:
            thresholds.append((float("inf"), sorted_scores[-1][0]))

        _category_thresholds = thresholds

        logger.info(
            "Category thresholds computed from cluster CSVs: %s",
            [(f"{t:.2f}", l) for t, l in thresholds if t != float("inf")],
        )

    except Exception as e:
        logger.error("Failed to compute category thresholds: %s", e)
        _category_thresholds = None


def _backup_artifacts(artifacts_path):
    """Backup current artifacts to a .bak folder."""
    backup_dir = artifacts_path + ".bak"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    if os.path.exists(artifacts_path):
        shutil.copytree(artifacts_path, backup_dir)
        logger.info("Artifacts backed up to %s", backup_dir)


def _restore_artifacts(artifacts_path):
    """Restore artifacts from .bak folder."""
    backup_dir = artifacts_path + ".bak"
    if not os.path.exists(backup_dir):
        logger.error("No backup found to restore from")
        return False

    # Remove failed artifacts and restore backup
    for fname in os.listdir(backup_dir):
        src = os.path.join(backup_dir, fname)
        dst = os.path.join(artifacts_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    logger.info("Artifacts restored from backup")
    return True


# ===================== #
#   VERSION CHECKING    #
# ===================== #


def _get_latest_wandb_version(artifacts_path):
    """
    Query W&B for the latest artifact version digest.
    Returns (digest, artifact) tuple or (None, None) if unavailable.
    """
    try:
        import wandb

        skip_wandb = os.getenv("SKIP_WANDB_DOWNLOAD", "false").lower() == "true"
        if skip_wandb:
            return None, None

        wandb_project = os.getenv("WANDB_PROJECT", "mindsync-model")
        wandb_entity = os.getenv("WANDB_ENTITY", None)
        artifact_name = "mindsync-model-smart"

        api = wandb.Api()
        if wandb_entity:
            artifact_path = f"{wandb_entity}/{wandb_project}/{artifact_name}:latest"
        else:
            artifact_path = f"{wandb_project}/{artifact_name}:latest"

        artifact = api.artifact(artifact_path, type="model")
        return artifact.digest, artifact

    except ImportError:
        logger.debug("wandb not installed, skipping version check")
        return None, None
    except Exception as e:
        logger.warning("Version check failed: %s", e)
        return None, None


def _download_wandb_artifact(artifact, artifacts_path):
    """
    Download a specific W&B artifact to the artifacts directory.
    Returns True on success.
    """
    try:
        from pathlib import Path

        artifact_dir = artifact.download(root=artifacts_path)
        logger.info("Downloaded artifact to: %s", artifact_dir)

        versioned_path = Path(artifact_dir)
        artifacts_root = Path(artifacts_path)
        preserve_files = ["healthy_cluster_avg.csv"]

        if versioned_path != artifacts_root:
            for file in versioned_path.glob("*.pkl"):
                shutil.copy2(file, artifacts_root / file.name)
            for file in versioned_path.glob("*.csv"):
                if file.name not in preserve_files:
                    shutil.copy2(file, artifacts_root / file.name)
            try:
                shutil.rmtree(versioned_path)
            except Exception as e:
                logger.warning(f"Failed to remove versioned path {versioned_path}: {e}")

        return True
    except Exception as e:
        logger.error("Failed to download artifact: %s", e)
        return False


def _check_for_updates():
    """
    Background task: check W&B for newer model version.
    If found, download, validate, and hot-swap. If broken, rollback.
    """
    global _current_version, _version_check_timer

    try:
        if _artifacts_path is None:
            return

        logger.info("Checking for model updates...")
        latest_digest, artifact = _get_latest_wandb_version(_artifacts_path)

        if latest_digest is None:
            logger.info("Could not reach W&B, skipping update check")
            return

        if latest_digest == _current_version:
            logger.info("Model is up to date (version: %s...)", _current_version[:8])
            return

        logger.info(
            f"New model version found: {latest_digest[:8]}... "
            f"(current: {(_current_version or 'none')[:8]}...)"
        )

        # Backup current working artifacts
        _backup_artifacts(_artifacts_path)

        # Download new version
        if not _download_wandb_artifact(artifact, _artifacts_path):
            logger.error("Download failed, keeping current version")
            _restore_artifacts(_artifacts_path)
            return

        # Try loading the new artifacts
        try:
            loaded = _load_artifacts(_artifacts_path)
        except Exception as e:
            logger.error("Failed to load new artifacts: %s", e)
            _restore_artifacts(_artifacts_path)
            return

        # Validate the new model
        if not _validate_model(loaded):
            logger.error("New model failed validation, reverting to previous version")
            _restore_artifacts(_artifacts_path)
            # Reload the restored artifacts
            try:
                loaded = _load_artifacts(_artifacts_path)
                _apply_loaded_artifacts(loaded)
            except Exception as e:
                logger.error("Failed to reload after revert: %s", e)
            return

        # New model is valid — hot-swap
        _apply_loaded_artifacts(loaded)
        _compute_category_thresholds()
        _current_version = latest_digest
        logger.info("Model updated successfully to version %s...", latest_digest[:8])

    except Exception as e:
        logger.error("Unexpected error during update check: %s", e)

    finally:
        # Schedule next check
        _schedule_version_check()


def _schedule_version_check():
    """Schedule the next version check."""
    global _version_check_timer

    if _version_check_timer is not None:
        _version_check_timer.cancel()

    _version_check_timer = threading.Timer(_VERSION_CHECK_INTERVAL, _check_for_updates)
    _version_check_timer.daemon = True
    _version_check_timer.start()
    logger.debug("Next version check in %s minutes", _VERSION_CHECK_INTERVAL // 60)


# ===================== #
#   APP INITIALIZATION  #
# ===================== #


def init_app(app):
    """Initialize ML model with the app."""
    global model, preprocessor, healthy_cluster_df, coefficients_df
    global _current_version, _artifacts_path, _flask_app

    logger.info("Starting ML model initialization")
    _artifacts_path = os.path.join(app.root_path, "..", "artifacts")
    _flask_app = app
    logger.info("Artifacts path: %s", _artifacts_path)

    # Try to download artifacts from W&B
    download_artifacts_from_wandb(_artifacts_path)

    # Load and validate all artifacts
    try:
        loaded = _load_artifacts(_artifacts_path)
        if _validate_model(loaded):
            _apply_loaded_artifacts(loaded)
            logger.info("All artifacts loaded and validated")
        else:
            logger.error("Model validation failed on initial load")
            # Still apply — better to have a model than none
            _apply_loaded_artifacts(loaded)
        _compute_category_thresholds()
    except Exception as e:
        logger.error("Failed to load artifacts: %s", e)

    # Record current version from W&B if available
    digest, _ = _get_latest_wandb_version(_artifacts_path)
    if digest:
        _current_version = digest
        logger.info("Current model version: %s...", _current_version[:8])
    else:
        logger.info("Running with local artifacts (no W&B version tracked)")

    # Start background version checker
    skip_wandb = os.getenv("SKIP_WANDB_DOWNLOAD", "false").lower() == "true"
    if not skip_wandb:
        _schedule_version_check()
        logger.info(
            f"Version auto-update enabled (every "
            f"{_VERSION_CHECK_INTERVAL // 60} minutes)"
        )
    else:
        logger.info("Version auto-update disabled (SKIP_WANDB_DOWNLOAD=true)")

    logger.info("ML model initialization complete")


# ===================== #
#  PREPROCESSING UTILS  #
# ===================== #


def clean_occupation_column(df):
    """
    Clean occupation column by combining rare categories.
    """
    df_copy = df.copy()
    if "occupation" in df_copy.columns:
        df_copy["occupation"] = df_copy["occupation"].replace(
            ["Unemployed", "Retired"], "Unemployed"
        )
    return df_copy


# ===================== #
#   CUSTOM RIDGE MODEL  #
# ===================== #


class LinearRegressionRidge(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        normalize=False,
        solver="closed_form",
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-4,
        verbose=False,
    ):
        """
        Ridge Regression with multiple solvers.

        Parameters:
        -----------
        alpha : float
            Regularization strength
        fit_intercept : bool
            Whether to fit intercept
        normalize : bool
            Whether to normalize features
        solver : str
            One of {'closed_form', 'gd', 'sgd', 'svd', 'cholesky', 'cg', 'sag'}
            Options:
            - closed_form: OLS closed-form solution (best for small-medium)
            - gd: Gradient Descent (better for large datasets)
            - sgd: Stochastic Gradient Descent (best for very large datasets)
            - svd: SVD (numerically stable, handles collinearity)
            - cholesky: Cholesky Decomposition (fast for positive-definite)
            - cg: Conjugate Gradient (memory efficient for sparse systems)
            - sag: Stochastic Average Gradient (fast convergence)
        learning_rate : float
            Learning rate for gradient descent solvers
        max_iter : int
            Maximum iterations for iterative solvers (gd, sgd, cg, sag)
        tol : float
            Convergence tolerance for iterative solvers
        verbose : bool
            Print updates during training
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = 0.0
        self.scaler_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Normalize features if requested
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Add intercept term (column of ones)
        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X

        n_samples, n_features = X_b.shape

        # --- Select Solver ---
        if self.solver == "closed_form":
            self._solve_closed_form(X_b, y, n_features)
        elif self.solver == "gd":
            self._solve_gradient_descent(X_b, y, n_samples, n_features)
        elif self.solver == "sgd":
            self._solve_sgd(X_b, y, n_samples, n_features)
        elif self.solver == "svd":
            self._solve_svd(X_b, y, n_features)
        elif self.solver == "cholesky":
            self._solve_cholesky(X_b, y, n_features)
        elif self.solver == "cg":
            self._solve_cg(X_b, y, n_features)
        elif self.solver == "sag":
            self._solve_sag(X_b, y, n_samples, n_features)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return self

    def _solve_closed_form(self, X, y, n_features):
        """Solves using (XᵀX + αI)⁻¹Xᵀy"""
        # Identity matrix for regularization
        # We don't penalize the intercept, so identity_matrix[0,0] = 0
        identity_matrix = np.eye(n_features)
        if self.fit_intercept:
            identity_matrix[0, 0] = 0

        try:
            # Calculate weights (theta)
            A = X.T @ X + self.alpha * identity_matrix
            b = X.T @ y
            weights = np.linalg.solve(A, b)

        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if solve fails (e.g., singular matrix)
            print("Warning: SVD (pseudo-inverse) fallback used.")
            weights = np.linalg.pinv(A) @ b

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_svd(self, X, y, n_features):
        """
        Solves Ridge via Singular Value Decomposition.

        Decomposes X = UΣVᵀ, then:
            w = V · diag(σᵢ / (σᵢ² + α)) · Uᵀy

        Advantages:
        - Numerically very stable
        - Handles multicollinearity gracefully
        - No matrix inversion needed
        """
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Build regularization penalty per singular value
        # For intercept: don't penalize the first component
        # We apply penalty uniformly here since SVD mixes columns;
        # the intercept-column contribution is naturally handled
        # by the decomposition structure.
        reg = np.zeros_like(sigma)
        for i in range(len(sigma)):
            reg[i] = sigma[i] / (sigma[i] ** 2 + self.alpha)

        # If fit_intercept, we skip penalizing intercept direction
        # SVD approach: compute d = diag(σ/(σ²+α))
        d = sigma / (sigma**2 + self.alpha)

        # weights = V^T^T · diag(d) · U^T · y = V · diag(d) · U^T · y
        weights = Vt.T @ (d * (U.T @ y))

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_cholesky(self, X, y, n_features):
        """
        Solves Ridge via Cholesky Decomposition.

        Decomposes A = (XᵀX + αI) = LLᵀ (lower triangular),
        then solves via forward + backward substitution:
            Lz = Xᵀy   (forward substitution)
            Lᵀw = z     (backward substitution)

        Advantages:
        - ~2x faster than general linear solve for positive-definite systems
        - Numerically stable for well-conditioned matrices
        """
        identity_matrix = np.eye(n_features)
        if self.fit_intercept:
            identity_matrix[0, 0] = 0  # Don't penalize intercept

        A = X.T @ X + self.alpha * identity_matrix
        b = X.T @ y

        try:
            # Cholesky decomposition: A = L @ L.T
            L = np.linalg.cholesky(A)

            # Forward substitution: solve L @ z = b
            z = np.linalg.solve(L, b)

            # Backward substitution: solve L.T @ weights = z
            weights = np.linalg.solve(L.T, z)

        except np.linalg.LinAlgError:
            # Fallback if matrix is not positive-definite
            if self.verbose:
                print(
                    "Warning: Cholesky failed (not positive-definite). "
                    "Falling back to np.linalg.solve."
                )
            weights = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_cg(self, X, y, n_features):
        """
        Solves Ridge via Conjugate Gradient (iterative method).

        Iteratively solves (XᵀX + αI)w = Xᵀy without forming the
        full matrix explicitly — uses matrix-vector products only.

        Advantages:
        - Memory efficient: O(n_features) instead of O(n_features²)
        - Excellent for large, sparse systems
        - Converges in at most n_features iterations (in exact arithmetic)
        """
        I_diag = np.ones(n_features)
        if self.fit_intercept:
            I_diag[0] = 0  # Don't penalize intercept

        # Right-hand side
        rhs = X.T @ y  # Xᵀy

        # Initialize weights
        weights = np.zeros(n_features)

        # r = b - Ax (residual), where A = XᵀX + αI
        Aw = X.T @ (X @ weights) + self.alpha * (I_diag * weights)
        r = rhs - Aw
        p = r.copy()  # search direction
        rs_old = r @ r  # r^T r

        for iteration in range(min(self.max_iter, n_features * 2)):
            # Matrix-vector product: A @ p
            Ap = X.T @ (X @ p) + self.alpha * (I_diag * p)

            # Step size
            pAp = p @ Ap
            if pAp < 1e-30:
                if self.verbose:
                    print(f"CG: pAp near zero at iteration {iteration}, stopping.")
                break
            alpha_cg = rs_old / pAp

            # Update solution and residual
            weights = weights + alpha_cg * p
            r = r - alpha_cg * Ap

            rs_new = r @ r

            # Check convergence
            if np.sqrt(rs_new) < self.tol:
                if self.verbose:
                    print(f"CG converged at iteration {iteration}")
                break

            # Update search direction
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

            if self.verbose and iteration % 100 == 0:
                print(
                    f"CG iteration {iteration}: residual norm = {np.sqrt(rs_new):.8f}"
                )

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_sag(self, X, y, n_samples, n_features):
        """
        Solves Ridge via Stochastic Average Gradient (SAG).

        Maintains a table of per-sample gradients and updates
        the running average at each step, giving:
        - Stochastic efficiency (one sample per step)
        - Linear convergence rate (like full gradient descent)

        Advantages:
        - Much faster than GD for large datasets
        - Linear convergence rate (better than SGD's sublinear)
        - Memory cost: O(n_samples × n_features) for gradient table
        """
        weights = np.zeros(n_features)

        # Table of per-sample gradients (initialized to zero)
        grad_table = np.zeros((n_samples, n_features))
        sum_gradients = np.zeros(n_features)  # Running sum of all gradients

        # Number of gradients seen so far (for averaging)
        seen = np.zeros(n_samples, dtype=bool)
        n_seen = 0

        for epoch in range(self.max_iter):
            prev_weights = weights.copy()

            for _ in range(n_samples):
                # Pick a random sample
                i = np.random.randint(n_samples)

                xi = X[i : i + 1]  # noqa: E203
                yi = y[i]

                # Compute new gradient for sample i
                pred = xi @ weights
                error = pred - yi
                new_grad = xi.flatten() * error

                # Update the sum: subtract old, add new
                sum_gradients = sum_gradients - grad_table[i] + new_grad
                grad_table[i] = new_grad

                # Track unique samples seen
                if not seen[i]:
                    seen[i] = True
                    n_seen += 1

                # Average gradient + L2 penalty
                avg_grad = sum_gradients / max(n_seen, 1)
                ridge_grad = self.alpha * weights
                if self.fit_intercept:
                    ridge_grad[0] = 0

                # Update weights
                weights = weights - self.learning_rate * (avg_grad + ridge_grad)

            # Check convergence at epoch level
            weight_change = np.linalg.norm(weights - prev_weights)
            if weight_change < self.tol:
                if self.verbose:
                    print(f"SAG converged at epoch {epoch}")
                break

            if self.verbose and epoch % 100 == 0:
                y_pred = X @ weights
                loss = np.sum((y_pred - y) ** 2) / (2 * n_samples) + (
                    self.alpha / 2
                ) * np.sum(weights[1:] ** 2 if self.fit_intercept else weights**2)
                print(
                    f"SAG Epoch {epoch}: Loss = {loss:.6f}, "
                    f"Weight change = {weight_change:.8f}"
                )

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_gradient_descent(self, X, y, n_samples, n_features):
        """Solves using batch Gradient Descent"""
        weights = np.zeros(n_features)
        y = y.reshape(-1, 1)  # Ensure y is a column vector
        weights = weights.reshape(-1, 1)  # Ensure weights is a column vector

        for iteration in range(self.max_iter):
            y_pred = X @ weights
            error = y_pred - y

            # Gradient calculation
            gradient = (X.T @ error) / n_samples

            # Add L2 penalty (Ridge)
            # Don't penalize bias term w[0]
            ridge_grad = self.alpha * weights
            if self.fit_intercept:
                ridge_grad[0] = 0

            gradient += ridge_grad

            # Update weights
            new_weights = weights - self.learning_rate * gradient

            # Check for convergence
            if np.linalg.norm(new_weights - weights) < self.tol:
                if self.verbose:
                    print(f"GD converged at iteration {iteration}")
                weights = new_weights
                break

            weights = new_weights

            if self.verbose and iteration % 100 == 0:
                loss = (np.sum(error**2) / (2 * n_samples)) + (
                    self.alpha / 2
                ) * np.sum(weights[1:] ** 2 if self.fit_intercept else weights**2)
                print(f"Epoch {iteration}: Loss = {loss:.6f}")

        weights = weights.flatten()
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_sgd(self, X, y, n_samples, n_features):
        """Solves using Stochastic Gradient Descent"""
        weights = np.zeros(n_features)

        for epoch in range(self.max_iter):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            prev_weights = np.copy(weights)

            for i in range(n_samples):
                xi = X_shuffled[i : i + 1]  # noqa: E203
                yi = y_shuffled[i : i + 1]  # noqa: E203

                y_pred = xi @ weights
                error = y_pred - yi

                # Gradient for single sample
                gradient = xi.T * error

                # L2 penalty
                ridge_grad = self.alpha * weights
                if self.fit_intercept:
                    ridge_grad[0] = 0

                # --- THIS IS THE FIX ---
                gradient += ridge_grad.reshape(-1, 1)

                # Update weights
                weights = weights - self.learning_rate * gradient.flatten()

            # Check for convergence (epoch level)
            if np.linalg.norm(weights - prev_weights) < self.tol:
                if self.verbose:
                    print(f"SGD converged at epoch {epoch}")
                break

            if self.verbose and epoch % 100 == 0:
                y_full_pred = X @ weights
                loss = (np.sum((y_full_pred - y) ** 2) / (2 * n_samples)) + (
                    self.alpha / 2
                ) * np.sum(weights[1:] ** 2 if self.fit_intercept else weights**2)
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _calculate_loss(self, X, y, weights, n_samples):
        """Helper for verbose GD/SGD loss"""
        y_pred = X @ weights
        mse_loss = np.sum((y_pred - y) ** 2) / (2 * n_samples)

        # L2 penalty term (excluding intercept)
        if self.fit_intercept:
            l2_loss = (self.alpha / 2) * np.sum(weights[1:] ** 2)
        else:
            l2_loss = (self.alpha / 2) * np.sum(weights**2)

        return mse_loss + l2_loss

    def _check_convergence(self, weights, prev_weights, iteration):
        """Helper for GD/SGD convergence check"""
        if np.linalg.norm(weights - prev_weights) < self.tol:
            if self.verbose:
                print(f"Converged at iteration {iteration}")
            return True
        return False

    def predict(self, X):
        X = np.array(X, dtype=np.float64)

        if self.normalize and self.scaler_ is not None:
            X = self.scaler_.transform(X)

        prediction = self.intercept_ + X @ self.coef_

        return prediction

    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# ===================== #
#   ANALYSIS FUNCTIONS  #
# ===================== #


def analyze_wellness_factors(user_df):
    """
    Analyze wellness factors by comparing healthy cluster vs user input.
    """
    logger.debug("Starting wellness factor analysis")
    if preprocessor is None or healthy_cluster_df is None or coefficients_df is None:
        logger.error("Cannot analyze wellness factors: required components not loaded")
        return None

    try:
        logger.debug("Preprocessing healthy cluster and user data")
        healthy_preprocessed = preprocessor.transform(healthy_cluster_df)
        user_preprocessed = preprocessor.transform(user_df)

        if hasattr(healthy_preprocessed, "toarray"):
            healthy_arr = healthy_preprocessed.toarray()[0]
            user_arr = user_preprocessed.toarray()[0]
        else:
            healthy_arr = healthy_preprocessed[0]
            user_arr = user_preprocessed[0]

        logger.debug("Analyzing %s wellness factors", len(coefficients_df))
        results = []
        for idx, row in coefficients_df.iterrows():
            if idx < len(healthy_arr):
                feature_name = row["Feature"]
                coefficient = row["Coefficient"]
                healthy_value = healthy_arr[idx]
                user_value = user_arr[idx]
                difference = healthy_value - user_value
                impact_score = difference * coefficient

                results.append(
                    {
                        "feature": feature_name,
                        "impact_score": float(impact_score),
                        "coefficient": float(coefficient),
                        "healthy_value": float(healthy_value),
                        "user_value": float(user_value),
                        "gap": float(difference),
                    }
                )

        areas_for_improvement = [r for r in results if r["impact_score"] > 0]
        strengths = [r for r in results if r["impact_score"] < 0]

        areas_for_improvement.sort(key=lambda x: x["impact_score"], reverse=True)
        strengths.sort(key=lambda x: abs(x["impact_score"]), reverse=True)

        logger.info(
            f"Wellness analysis complete: {len(areas_for_improvement)}"
            f"improvements, {len(strengths)} strengths"
        )
        logger.debug(
            "Top improvement area: %s",
            areas_for_improvement[0]["feature"] if areas_for_improvement else "None",
        )

        return {
            "areas_for_improvement": areas_for_improvement[:5],
            "strengths": strengths[:5],
        }

    except Exception as e:
        logger.error("Error in wellness analysis: %s", e)
        return None


def categorize_mental_health_score(prediction_score):
    """Categorize mental health score using dynamic thresholds from cluster averages."""
    prediction_score = float(prediction_score)
    capped_score = min(max(prediction_score, 0), 100)

    if _category_thresholds is not None:
        # Use dynamic thresholds computed from cluster averages
        category = _category_thresholds[-1][1]  # default to last (highest) category
        for upper_bound, label in _category_thresholds:
            if capped_score <= upper_bound:
                category = label
                break
    else:
        # Fallback to static thresholds if not yet computed
        if capped_score <= 12:
            category = "dangerous"
        elif capped_score <= 28.6:
            category = "not healthy"
        elif capped_score <= 61.4:
            category = "average"
        else:
            category = "healthy"

    logger.debug(
        f"Mental health score {prediction_score:.2f} categorized as '{category}'"
    )
    return category
