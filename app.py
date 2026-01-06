import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import numpy as np

# --- 1. ADD ALL PREPROCESSING IMPORTS ---
# Pickle needs these libraries loaded to reconstruct the pipeline
from sklearn.base import BaseEstimator, RegressorMixin 
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    PolynomialFeatures, 
    PowerTransformer, 
    FunctionTransformer
)


app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join('artifacts', 'model.pkl')

# --- 2. DEFINE THE CUSTOM CLEANER FUNCTION ---
# This must exist exactly as it did in the notebook
def clean_occupation_column(df):
    """
    Mendeteksi kolom 'occupation' dan menggabungkan
    kategori jarang (Unemployed, Retired) menjadi 'Other'.
    """
    df_copy = df.copy()
    if 'occupation' in df_copy.columns:
        df_copy['occupation'] = df_copy['occupation'].replace(
            ['Unemployed', 'Retired'], 'Unemployed'
        )
    return df_copy

# --- 3. DEFINE THE CUSTOM MODEL CLASS ---
class LinearRegressionRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, 
                 solver='closed_form', learning_rate=0.01, max_iter=1000, 
                 tol=1e-4, verbose=False):
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
        solver : str, one of {'closed_form', 'gd', 'sgd'}
            - 'closed_form': OLS closed-form solution (best for small-medium datasets)
            - 'gd': Gradient Descent (better for large datasets)
            - 'sgd': Stochastic Gradient Descent (best for very large datasets)
        learning_rate : float
            Learning rate for gradient descent solvers
        max_iter : int
            Maximum iterations for gradient descent
        tol : float
            Convergence tolerance for gradient descent
        verbose : bool
            Print updates during gradient descent
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
        if self.solver == 'closed_form':
            self._solve_closed_form(X_b, y, n_features)
        elif self.solver == 'gd':
            self._solve_gradient_descent(X_b, y, n_samples, n_features)
        elif self.solver == 'sgd':
            self._solve_sgd(X_b, y, n_samples, n_features)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        return self

    def _solve_closed_form(self, X, y, n_features):
        """Solves using (XᵀX + αI)⁻¹Xᵀy"""
        # Identity matrix for regularization
        # We don't penalize the intercept, so I[0,0] = 0
        I = np.eye(n_features)
        if self.fit_intercept:
            I[0, 0] = 0 
            
        try:
            # Calculate weights (theta)
            A = X.T @ X + self.alpha * I
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

    def _solve_gradient_descent(self, X, y, n_samples, n_features):
        """Solves using batch Gradient Descent"""
        weights = np.zeros(n_features)
        y = y.reshape(-1, 1) # Ensure y is a column vector
        weights = weights.reshape(-1, 1) # Ensure weights is a column vector

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
                if self.verbose: print(f"GD converged at iteration {iteration}")
                weights = new_weights
                break
            
            weights = new_weights
            
            if self.verbose and iteration % 100 == 0:
                loss = (np.sum(error**2) / (2 * n_samples)) + (self.alpha / 2) * np.sum(weights[1:]**2 if self.fit_intercept else weights**2)
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
                xi = X_shuffled[i:i+1] # Keep 2D shape
                yi = y_shuffled[i:i+1]
                
                y_pred = xi @ weights
                error = y_pred - yi
                
                # Gradient for single sample
                gradient = (xi.T * error)
                
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
                if self.verbose: print(f"SGD converged at epoch {epoch}")
                break
                
            if self.verbose and epoch % 100 == 0:
                y_full_pred = X @ weights
                loss = (np.sum((y_full_pred - y)**2) / (2 * n_samples)) + (self.alpha / 2) * np.sum(weights[1:]**2 if self.fit_intercept else weights**2)
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
            l2_loss = (self.alpha / 2) * np.sum(weights[1:]**2)
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
        
        return self.intercept_ + X @ self.coef_
    
    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# --- 4. LOAD MODEL ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

# Load model inside a Try/Except block to catch the specific error
try:
    model = load_model()
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    # This print is crucial for debugging
    print(f"CRITICAL ERROR loading model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "message": "Model API is running."
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({
            "error": "Model failed to load. Check server logs for details."
        }), 500

    try:
        json_input = request.get_json()
        
        # Handle dict vs list input
        if isinstance(json_input, dict):
            df = pd.DataFrame([json_input])
        else:
            df = pd.DataFrame(json_input)

        prediction = model.predict(df)
        
        return jsonify({
            "prediction": prediction.tolist(),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)