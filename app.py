import pickle
import pandas as pd
from flask import Flask, request, jsonify
import os
import numpy as np
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uuid
import threading
from datetime import datetime
import valkey

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

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
PREPROCESSOR_PATH = os.path.join('artifacts', 'preprocessor.pkl')
COEFFICIENTS_PATH = os.path.join('artifacts', 'model_coefficients.csv')
HEALTHY_CLUSTER_PATH = os.path.join('artifacts', 'healthy_cluster_avg.csv')

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

# --- 4. LOAD MODEL AND PREPROCESSOR ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

def load_preprocessor():
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
    
    with open(PREPROCESSOR_PATH, 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor

def load_healthy_cluster():
    if not os.path.exists(HEALTHY_CLUSTER_PATH):
        raise FileNotFoundError(f"Healthy cluster file not found at {HEALTHY_CLUSTER_PATH}")
    
    df = pd.read_csv(HEALTHY_CLUSTER_PATH)
    return df

def load_coefficients():
    if not os.path.exists(COEFFICIENTS_PATH):
        raise FileNotFoundError(f"Coefficients file not found at {COEFFICIENTS_PATH}")
    
    df = pd.read_csv(COEFFICIENTS_PATH)
    return df

# Load model inside a Try/Except block to catch the specific error
try:
    model = load_model()
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    # This print is crucial for debugging
    print(f"CRITICAL ERROR loading model: {e}")
    model = None

# Load preprocessor and supporting data
try:
    preprocessor = load_preprocessor()
    healthy_cluster_df = load_healthy_cluster()
    coefficients_df = load_coefficients()
    print(f"Successfully loaded preprocessor and analysis data")
except Exception as e:
    print(f"WARNING: Could not load analysis components: {e}")
    preprocessor = None
    healthy_cluster_df = None
    coefficients_df = None

def analyze_wellness_factors(user_df):
    """
    Analyze wellness factors by comparing healthy cluster vs user input.
    Returns top positive and negative contributing factors.
    """
    if preprocessor is None or healthy_cluster_df is None or coefficients_df is None:
        return None
    
    try:
        # Preprocess both inputs
        healthy_preprocessed = preprocessor.transform(healthy_cluster_df)
        user_preprocessed = preprocessor.transform(user_df)
        
        # Convert to arrays if sparse
        if hasattr(healthy_preprocessed, 'toarray'):
            healthy_arr = healthy_preprocessed.toarray()[0]
            user_arr = user_preprocessed.toarray()[0]
        else:
            healthy_arr = healthy_preprocessed[0]
            user_arr = user_preprocessed[0]
        
        # Calculate impact scores
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
        
        # Separate and sort
        areas_for_improvement = [r for r in results if r['impact_score'] > 0]
        strengths = [r for r in results if r['impact_score'] < 0]
        
        areas_for_improvement.sort(key=lambda x: x['impact_score'], reverse=True)
        strengths.sort(key=lambda x: abs(x['impact_score']), reverse=True)
        
        return {
            'areas_for_improvement': areas_for_improvement[:5],  # Top 5 gaps to address
            'strengths': strengths[:5]   # Top 5 factors user excels at
        }
    
    except Exception as e:
        print(f"Error in wellness factor analysis: {e}")
        return None

def categorize_mental_health_score(prediction_score):
    prediction_score = float(prediction_score)
    if prediction_score <= 12:
        category = "dangerous"
    elif prediction_score <= 28.6:
        category = "not healthy"
    elif prediction_score <= 61.4:
        category = "average"
    else:
        category = "healthy"
    return category

# Gemini Advisor
def get_ai_advice(prediction_score, category, wellness_analysis_result):
    # Extract Top Factors
    top_factors_list = []
    if wellness_analysis_result and 'areas_for_improvement' in wellness_analysis_result:
        # Ambil Top 3
        top_factors_list = [item['feature'] for item in wellness_analysis_result['areas_for_improvement'][:3]]
    
    # Format string factor
    if top_factors_list:
        factors_inline_str = ", ".join(top_factors_list)
        factors_bullet_list = "\n".join([f"- {f}" for f in top_factors_list])
    else:
        factors_inline_str = "General Wellness"
        factors_bullet_list = "- General Wellness"

    # Context Resources
    trusted_sources_context = """
    - https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/5-ways-to-get-better-sleep
    - https://www.aoa.org/healthy-eyes/eye-and-vision-conditions/computer-vision-syndrome
    - https://www.sleepfoundation.org/sleep-hygiene
    - https://www.cdc.gov/sleep/about/index.html
    - https://www.apa.org/topics/stress/tips
    - https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health
    - https://www.who.int/news-room/fact-sheets/detail/physical-activity
    - https://www.nia.nih.gov/health/brain-health/cognitive-health-and-older-adults
    - https://www.health.harvard.edu/staying-healthy/the-health-benefits-of-strong-relationships
    - https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-minute-boost-your-health-and-productivity-with-activity-snacks/
    - https://youtu.be/dlgCJd1cfy8?si=mmk2X8vvUGjtvWBJ
    """

    # Prompt Engineering
    prompt = f"""
    Role: You are 'MindSync', a mental health AI advisor.
    
    User Context:
    - Risk Level: "{category}" (Score: {prediction_score})
    - Main Struggles: {factors_bullet_list}
    
    Task:
    Generate advice and return it STRICTLY as a JSON object with the following structure:
    
    {{
        "description": "String (warm empathy paragraph)",
        "factors": {{
            "FACTOR_NAME_1": {{
                "advices": ["String 1", "String 2", "String 3"],
                "references": ["URL 1", "URL 2"]
            }},
            "FACTOR_NAME_2": {{
                ...
            }}
        }}
    }}

    Detailed Requirements:

    1. "description": 
       - Write a warm, validating paragraph based on Risk Level "{category}".
       - Do NOT explicitly state the category name.
       - Acknowledge that dealing with {factors_inline_str} is challenging.

    2. "factors":
       - Create a dictionary key for EACH item in the "Main Struggles" list: {factors_inline_str}.
       - For each factor, provide:
         a. "advices" (Array of Strings): Exactly 3 actionable tips specific to that factor.
         b. "references" (Array of Strings): Select 1 to 3 relevant URLs specifically for this factor from the list below:
            {trusted_sources_context}
            (If no link matches perfectly, use a general mental health link).
    
    Tone: Professional, warm, non-judgmental.
    Language: English (Standard US).
    """

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Gemini Error: {e}")
        # Fallback Structure
        return {
            "description": "We encountered a temporary issue generating your personalized plan.",
            "factors": {}
        }

# Valkey client for storing prediction results
try:
    valkey_client = valkey.Valkey(
        host=os.getenv('VALKEY_HOST', 'localhost'),
        port=int(os.getenv('VALKEY_PORT', 6379)),
        db=int(os.getenv('VALKEY_DB', 0)),
        username=os.getenv('VALKEY_USERNAME'),
        password=os.getenv('VALKEY_PASSWORD'),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True
    )
    # Test connection
    valkey_client.ping()
    print("Successfully connected to Valkey")
except Exception as e:
    print(f"WARNING: Could not connect to Valkey: {e}")
    print("Application will continue but prediction storage may fail")
    valkey_client = None

# Helper functions for Valkey storage
def store_prediction(prediction_id, prediction_data):
    """Store prediction data in Valkey with 24-hour expiration"""
    if valkey_client is None:
        raise ConnectionError("Valkey client is not initialized")
    
    try:
        key = f"prediction:{prediction_id}"
        valkey_client.setex(
            key,
            86400,  # 24 hours in seconds
            json.dumps(prediction_data)
        )
    except valkey.exceptions.ConnectionError as e:
        print(f"Valkey connection error in store_prediction: {e}")
        raise ConnectionError(f"Failed to store prediction: Valkey connection error")
    except valkey.exceptions.TimeoutError as e:
        print(f"Valkey timeout error in store_prediction: {e}")
        raise TimeoutError(f"Failed to store prediction: Valkey timeout")
    except Exception as e:
        print(f"Unexpected error in store_prediction: {e}")
        raise RuntimeError(f"Failed to store prediction: {str(e)}")

def fetch_prediction(prediction_id):
    """Fetch prediction data from Valkey"""
    if valkey_client is None:
        raise ConnectionError("Valkey client is not initialized")
    
    try:
        key = f"prediction:{prediction_id}"
        data = valkey_client.get(key)
        if data:
            return json.loads(data)
        return None
    except valkey.exceptions.ConnectionError as e:
        print(f"Valkey connection error in fetch_prediction: {e}")
        raise ConnectionError(f"Failed to fetch prediction: Valkey connection error")
    except valkey.exceptions.TimeoutError as e:
        print(f"Valkey timeout error in fetch_prediction: {e}")
        raise TimeoutError(f"Failed to fetch prediction: Valkey timeout")
    except json.JSONDecodeError as e:
        print(f"JSON decode error in fetch_prediction: {e}")
        raise ValueError(f"Failed to parse prediction data")
    except Exception as e:
        print(f"Unexpected error in fetch_prediction: {e}")
        raise RuntimeError(f"Failed to fetch prediction: {str(e)}")

@app.route('/')
def home():
    return jsonify({
        "status": "active",     
        "message": "Model API is running."
    })

def process_prediction(prediction_id, json_input):
    """Background task untuk memproses prediction"""
    try:
        # Handle dict vs list input
        if isinstance(json_input, dict):
            df = pd.DataFrame([json_input])
        else:
            df = pd.DataFrame(json_input)

        # Make prediction
        prediction = model.predict(df)
        prediction_score = float(prediction[0])
        
        # Analyze wellness factors
        wellness_analysis = analyze_wellness_factors(df)
        
        # Categorize the mental health score
        mental_health_category = categorize_mental_health_score(prediction_score)
        
        # Get AI advice
        ai_advice = get_ai_advice(prediction_score, mental_health_category, wellness_analysis)
        
        # Update store dengan hasil
        try:
            existing_data = fetch_prediction(prediction_id)
            store_prediction(prediction_id, {
                "status": "ready",
                "result": {
                    "prediction_score": prediction_score,
                    "health_level": mental_health_category,
                    "wellness_analysis": wellness_analysis,
                    "advice": ai_advice
                },
                "created_at": existing_data["created_at"] if existing_data else datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            })
        except (ConnectionError, TimeoutError, RuntimeError) as storage_error:
            print(f"Failed to store prediction result: {storage_error}")
            
    except Exception as e:
        # Update store dengan error
        try:
            existing_data = fetch_prediction(prediction_id)
            store_prediction(prediction_id, {
                "status": "error",
                "error": str(e),
                "created_at": existing_data["created_at"] if existing_data else datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat()
            })
        except (ConnectionError, TimeoutError, RuntimeError) as storage_error:
            print(f"Failed to store error status: {storage_error}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({
            "error": "Model failed to load. Check server logs for details."
        }), 500

    try:
        json_input = request.get_json()
        
        # Generate unique prediction_id
        prediction_id = str(uuid.uuid4())
        
        # Initialize status sebagai processing
        try:
            store_prediction(prediction_id, {
                "status": "processing",
                "result": None,
                "created_at": datetime.now().isoformat()
            })
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            return jsonify({
                "error": "Storage service unavailable. Please try again later.",
                "details": str(e),
                "status": "error"
            }), 503
        
        # Start background processing
        thread = threading.Thread(
            target=process_prediction,
            args=(prediction_id, json_input)
        )
        thread.daemon = True
        thread.start()
        
        # Return prediction_id immediately
        return jsonify({
            "prediction_id": prediction_id,
            "status": "processing",
            "message": "Prediction is being processed. Use /result/<prediction_id> to check status."
        }), 202

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.route('/result/<prediction_id>', methods=['GET'])
def get_result(prediction_id):
    """Endpoint untuk polling hasil prediction"""
    try:
        prediction_data = fetch_prediction(prediction_id)
    except (ConnectionError, TimeoutError, RuntimeError) as e:
        return jsonify({
            "status": "error",
            "message": "Storage service unavailable. Please try again later.",
            "details": str(e)
        }), 503
    
    if not prediction_data:
        return jsonify({
            "status": "not_found",
            "message": "Prediction ID not found"
        }), 404
    
    status = prediction_data["status"]
    
    if status == "processing":
        return jsonify({
            "status": "processing",
            "message": "Prediction is still being processed. Please try again in a moment."
        }), 202
    
    elif status == "ready":
        return jsonify({
            "status": "ready",
            "result": prediction_data["result"],
            "created_at": prediction_data["created_at"],
            "completed_at": prediction_data["completed_at"]
        }), 200
    
    elif status == "error":
        return jsonify({
            "status": "error",
            "error": prediction_data["error"],
            "created_at": prediction_data["created_at"],
            "completed_at": prediction_data.get("completed_at")
        }), 500

@app.route('/advice', methods=['POST'])
def advice():
    """Legacy endpoint - masih bisa digunakan untuk backward compatibility"""
    try:
        json_input = request.get_json()
        
        prediction_score = json_input.get('prediction_score')
        if prediction_score is None:
            pred_list = json_input.get('prediction')
            if pred_list and isinstance(pred_list, list):
                prediction_score = pred_list[0]
            else:
                prediction_score = pred_list

        category = json_input.get('mental_health_category')
        analysis = json_input.get('wellness_analysis')

        if prediction_score is None or category is None or analysis is None:
            return jsonify({"error": "Missing inputs from /predict result", "status": "error"}), 400

        ai_advice = get_ai_advice(prediction_score, category, analysis)

        return jsonify({"ai_advice": ai_advice, "status": "success"})

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)