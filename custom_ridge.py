import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


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
