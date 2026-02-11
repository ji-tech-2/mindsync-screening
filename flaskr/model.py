"""
Machine Learning Model and Preprocessing
Custom Ridge Regression Implementation
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

# Global variables for model components
model = None
preprocessor = None
healthy_cluster_df = None
coefficients_df = None

def init_app(app):
    """Initialize ML model with the app."""
    global model, preprocessor, healthy_cluster_df, coefficients_df
    
    artifacts_path = os.path.join(app.root_path, '..', 'artifacts')

    # Register custom objects under __main__ for pickle compatibility
    sys.modules['__main__'].clean_occupation_column = clean_occupation_column
    sys.modules['__main__'].LinearRegressionRidge = LinearRegressionRidge
    
    # Load model
    try:
        model_path = os.path.join(artifacts_path, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
    
    # Load preprocessor
    try:
        preprocessor_path = os.path.join(artifacts_path, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"✅ Preprocessor loaded")
    except Exception as e:
        print(f"❌ Failed to load preprocessor: {e}")
        preprocessor = None
    
    # Load healthy cluster data
    try:
        healthy_path = os.path.join(artifacts_path, 'healthy_cluster_avg.csv')
        healthy_cluster_df = pd.read_csv(healthy_path)
        print(f"✅ Healthy cluster data loaded")
    except Exception as e:
        print(f"❌ Failed to load healthy cluster: {e}")
        healthy_cluster_df = None
    
    # Load coefficients
    try:
        coef_path = os.path.join(artifacts_path, 'model_coefficients.csv')
        coefficients_df = pd.read_csv(coef_path)
        print(f"✅ Coefficients loaded")
    except Exception as e:
        print(f"❌ Failed to load coefficients: {e}")
        coefficients_df = None

# ===================== #
#  PREPROCESSING UTILS  #
# ===================== #

def clean_occupation_column(df):
    """
    Clean occupation column by combining rare categories.
    """
    df_copy = df.copy()
    if 'occupation' in df_copy.columns:
        df_copy['occupation'] = df_copy['occupation'].replace(
            ['Unemployed', 'Retired'], 'Unemployed'
        )
    return df_copy

# ===================== #
#   CUSTOM RIDGE MODEL  #
# ===================== #

class LinearRegressionRidge(BaseEstimator, RegressorMixin):
    """
    Ridge Regression with multiple solvers.
    """
    
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, 
                 solver='closed_form', learning_rate=0.01, max_iter=1000, 
                 tol=1e-4, verbose=False):
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
        
        if self.normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X

        n_samples, n_features = X_b.shape
        
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
        """Closed-form solution: (X'X + αI)^-1 X'y"""
        I = np.eye(n_features)
        if self.fit_intercept:
            I[0, 0] = 0
        
        try:
            A = X.T @ X + self.alpha * I
            b = X.T @ y
            weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Warning: Using pseudo-inverse fallback")
            weights = np.linalg.pinv(A) @ b
        
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_gradient_descent(self, X, y, n_samples, n_features):
        """Batch Gradient Descent"""
        weights = np.zeros(n_features)
        y = y.reshape(-1, 1)
        weights = weights.reshape(-1, 1)

        for iteration in range(self.max_iter):
            y_pred = X @ weights
            error = y_pred - y
            
            gradient = (X.T @ error) / n_samples
            ridge_grad = self.alpha * weights
            if self.fit_intercept:
                ridge_grad[0] = 0
            
            gradient += ridge_grad
            new_weights = weights - self.learning_rate * gradient
            
            if np.linalg.norm(new_weights - weights) < self.tol:
                if self.verbose:
                    print(f"GD converged at iteration {iteration}")
                weights = new_weights
                break
            
            weights = new_weights
            
            if self.verbose and iteration % 100 == 0:
                loss = (np.sum(error**2) / (2 * n_samples)) + \
                       (self.alpha / 2) * np.sum(weights[1:]**2 if self.fit_intercept else weights**2)
                print(f"Epoch {iteration}: Loss = {loss:.6f}")
        
        weights = weights.flatten()
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

    def _solve_sgd(self, X, y, n_samples, n_features):
        """Stochastic Gradient Descent"""
        weights = np.zeros(n_features)
        
        for epoch in range(self.max_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            prev_weights = np.copy(weights)
            
            for i in range(n_samples):
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                
                y_pred = xi @ weights
                error = y_pred - yi
                
                gradient = (xi.T * error)
                ridge_grad = self.alpha * weights
                if self.fit_intercept:
                    ridge_grad[0] = 0
                
                gradient += ridge_grad.reshape(-1, 1)
                weights = weights - self.learning_rate * gradient.flatten()

            if np.linalg.norm(weights - prev_weights) < self.tol:
                if self.verbose:
                    print(f"SGD converged at epoch {epoch}")
                break
            
            if self.verbose and epoch % 100 == 0:
                y_full_pred = X @ weights
                loss = (np.sum((y_full_pred - y)**2) / (2 * n_samples)) + \
                       (self.alpha / 2) * np.sum(weights[1:]**2 if self.fit_intercept else weights**2)
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.coef_ = weights

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

# ===================== #
#   ANALYSIS FUNCTIONS  #
# ===================== #

def analyze_wellness_factors(user_df):
    """
    Analyze wellness factors by comparing healthy cluster vs user input.
    """
    if preprocessor is None or healthy_cluster_df is None or coefficients_df is None:
        return None
    
    try:
        healthy_preprocessed = preprocessor.transform(healthy_cluster_df)
        user_preprocessed = preprocessor.transform(user_df)
        
        if hasattr(healthy_preprocessed, 'toarray'):
            healthy_arr = healthy_preprocessed.toarray()[0]
            user_arr = user_preprocessed.toarray()[0]
        else:
            healthy_arr = healthy_preprocessed[0]
            user_arr = user_preprocessed[0]
        
        results = []
        for idx, row in coefficients_df.iterrows():
            if idx < len(healthy_arr):
                feature_name = row['Feature']
                coefficient = row['Coefficient']
                healthy_value = healthy_arr[idx]
                user_value = user_arr[idx]
                difference = healthy_value - user_value
                impact_score = difference * coefficient
                
                results.append({
                    'feature': feature_name,
                    'impact_score': float(impact_score),
                    'coefficient': float(coefficient),
                    'healthy_value': float(healthy_value),
                    'user_value': float(user_value),
                    'gap': float(difference)
                })
        
        areas_for_improvement = [r for r in results if r['impact_score'] > 0]
        strengths = [r for r in results if r['impact_score'] < 0]
        
        areas_for_improvement.sort(key=lambda x: x['impact_score'], reverse=True)
        strengths.sort(key=lambda x: abs(x['impact_score']), reverse=True)
        
        return {
            'areas_for_improvement': areas_for_improvement[:5],
            'strengths': strengths[:5]
        }
    
    except Exception as e:
        print(f"Error in wellness analysis: {e}")
        return None

def categorize_mental_health_score(prediction_score):
    """Categorize mental health score."""
    prediction_score = float(prediction_score)
    capped_score = min(max(prediction_score, 0), 100)

    if capped_score <= 12:
        return "dangerous"
    elif capped_score <= 28.6:
        return "not healthy"
    elif capped_score <= 61.4:
        return "average"
    else:
        return "healthy"
