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
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

# --- ADD ALL PREPROCESSING IMPORTS ---
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
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configuration
MODEL_PATH = os.path.join('artifacts', 'model.pkl')
PREPROCESSOR_PATH = os.path.join('artifacts', 'preprocessor.pkl')
COEFFICIENTS_PATH = os.path.join('artifacts', 'model_coefficients.csv')
HEALTHY_CLUSTER_PATH = os.path.join('artifacts', 'healthy_cluster_avg.csv')

# ===================== #
#    DATABASE MODELS    #
# ===================== #

# 1. PREDICTIONS 
class Predictions(db.Model):
    __tablename__ = 'PREDICTIONS' # Sesuai ERD
    pred_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), nullable=True)
    guest_id = db.Column(UUID(as_uuid=True), nullable=True)
    pred_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Input Data
    screen_time = db.Column(db.Float)
    work_screen = db.Column(db.Float)
    leisure_screen = db.Column(db.Float)
    sleep_hours = db.Column(db.Float)
    sleep_quality = db.Column(db.Integer)
    stress_level = db.Column(db.Float)
    productivity = db.Column(db.Float)
    exercise = db.Column(db.Integer)
    social = db.Column(db.Float)
    
    # Output Data
    pred_score = db.Column(db.Float) 
    
    # Relation (PRED_DETAILS)
    details = db.relationship('PredDetails', backref='prediction', lazy=True, cascade="all, delete-orphan")

# 2. PRED_DETAILS
class PredDetails(db.Model):
    __tablename__ = 'PRED_DETAILS'
    detail_id = db.Column(db.Integer, primary_key=True)
    pred_id = db.Column(UUID(as_uuid=True), db.ForeignKey('PREDICTIONS.pred_id'), nullable=False)
    factor_name = db.Column(db.String(100))
    impact_score = db.Column(db.Float)
    
    # Relation (ADVICES & REFERENCES)
    advices = db.relationship('Advices', backref='detail', lazy=True, cascade="all, delete-orphan")
    references = db.relationship('References', backref='detail', lazy=True, cascade="all, delete-orphan")

# 3. TABEL ADVICES 
class Advices(db.Model):
    __tablename__ = 'ADVICES' 
    advice_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(db.Integer, db.ForeignKey('PRED_DETAILS.detail_id'), nullable=False)
    advice_text = db.Column(db.Text)

# 4. TABEL REFERENCES
class References(db.Model):
    __tablename__ = 'REFERENCES'
    ref_id = db.Column(db.Integer, primary_key=True)
    detail_id = db.Column(db.Integer, db.ForeignKey('PRED_DETAILS.detail_id'), nullable=False)
    reference_link = db.Column(db.String(500))

# Create Tables if not exist
with app.app_context():
    # db.drop_all() # Uncomment jika ingin reset
    db.create_all()

# --- DEFINE THE CUSTOM CLEANER FUNCTION ---
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

# --- DEFINE THE CUSTOM MODEL CLASS ---
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

# Valkey client for storing prediction results - REQUIRED
print("Connecting to Valkey...")
valkey_url = os.getenv('VALKEY_URL', 'redis://localhost:6379')

try:
    print(f"Valkey URL: {valkey_url}")

    valkey_client = valkey.from_url(
        valkey_url,
        socket_connect_timeout=5,
        retry_on_timeout=True,
        socket_keepalive=True,
        health_check_interval=10
    )
    # Test connection
    valkey_client.ping()
    print("✅ Successfully connected to Valkey")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not connect to Valkey: {e}")
    print(f"❌ Application requires Valkey to be running at {valkey_url}")
    print("❌ Application will exit.")
    raise SystemExit(f"Failed to connect to Valkey: {e}")

# Helper functions for Valkey storage
def store_prediction(prediction_id, prediction_data):
    """Store prediction data with 24-hour expiration in Valkey"""
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

def update_prediction(prediction_id, update_data):
    """Update existing prediction data in Valkey"""
    key = f"prediction:{prediction_id}"
    existing = valkey_client.get(key)

    if not existing:
        raise KeyError("Prediction not found")

    data = json.loads(existing)
    data.update(update_data)

    valkey_client.setex(key, 86400, json.dumps(data))

def process_prediction(prediction_id, json_input, created_at, app_instance):
    try:
        if isinstance(json_input, dict):
            df = pd.DataFrame([json_input])
        else:
            df = pd.DataFrame(json_input)

        # FAST PART: Prediction & Analysis 
        prediction = model.predict(df)
        prediction_score = float(prediction[0])
        
        wellness_analysis = analyze_wellness_factors(df)
        mental_health_category = categorize_mental_health_score(prediction_score)
        
        # Result score
        try:
            store_prediction(prediction_id, {
                "status": "partial",
                "result": {
                    "prediction_score": prediction_score,
                    "health_level": mental_health_category,
                    "wellness_analysis": wellness_analysis,
                    "advice": None
                },
                "created_at": created_at if created_at else datetime.now().isoformat()
            })
            print(f"📊 Partial result ready for {prediction_id}")
        except Exception as partial_error:
            print(f"Failed to store partial result: {partial_error}")
        
        # SLOW PART: Gemini AI
        ai_advice = get_ai_advice(prediction_score, mental_health_category, wellness_analysis)
        
        # Final update
        try:
            update_prediction(prediction_id, {
                "status": "ready",
                "result": {
                    "prediction_score": prediction_score,
                    "health_level": mental_health_category,
                    "wellness_analysis": wellness_analysis,
                    "advice": ai_advice
                },
                "completed_at": datetime.now().isoformat()
            })
            print(f"✅ Full result ready for {prediction_id}")
        except Exception as update_error:
            print(f"Failed to update with advice: {update_error}")
            
        # Save to PostgreSQL database
        save_to_db(app_instance, prediction_id, json_input, prediction_score, wellness_analysis, ai_advice)

    except Exception as e:
        print(f"❌ Error processing {prediction_id}: {e}")
        # If fatal error, update status cache to 'error'
        try:
            update_prediction(prediction_id, {
                "status": "error",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
        except Exception as storage_error:
            print(f"Failed to store error status: {storage_error}")

def save_to_db(app_instance, prediction_id, json_input, prediction_score, wellness_analysis, ai_advice):
    with app_instance.app_context():
        try:
            u_id = uuid.UUID(json_input.get('user_id')) if json_input.get('user_id') else None
            
            new_pred = Predictions(
                pred_id=uuid.UUID(prediction_id),
                user_id=u_id,
                
                screen_time=float(json_input.get('screen_time_hours', 0)),
                work_screen=float(json_input.get('work_screen_hours', 0)),
                leisure_screen=float(json_input.get('leisure_screen_hours', 0)),
                sleep_hours=float(json_input.get('sleep_hours', 0)),
                stress_level=float(json_input.get('stress_level_0_10', 0)),
                productivity=float(json_input.get('productivity_0_100', 0)),
                social=float(json_input.get('social_hours_per_week', 0)),
                
                sleep_quality=int(json_input.get('sleep_quality_1_5', 0)),
                exercise=int(json_input.get('exercise_minutes_per_week', 0)),
                
                pred_score=prediction_score
            )
            db.session.add(new_pred)
            db.session.flush()  # Generate ID
            
            if wellness_analysis:
                for item in wellness_analysis.get('areas_for_improvement', []):
                    fname = item['feature']
                    
                    detail = PredDetails(
                        pred_id=new_pred.pred_id,
                        factor_name=fname,
                        impact_score=float(item['impact_score'])
                    )
                    db.session.add(detail)
                    db.session.flush()
                    
                    factor_data = ai_advice.get('factors', {}).get(fname, {})
                    
                    # Advices
                    for tip in factor_data.get('advices', []):
                        db.session.add(Advices(
                            detail_id=detail.detail_id,
                            advice_text=tip
                        ))
                        
                    # References
                    for ref in factor_data.get('references', []):
                        db.session.add(References(
                            detail_id=detail.detail_id,
                            reference_link=ref
                        ))

            db.session.commit()
            print(f"💾 SQL Save Completed for {prediction_id}")
            
        except Exception as sql_error:
            db.session.rollback()
            print(f"⚠️ SQL Save Failed: {sql_error}")
            return False
          
def read_from_db(prediction_id=None, user_id=None):
    try:
        if prediction_id:
            pred = Predictions.query.filter_by(pred_id=uuid.UUID(prediction_id)).first()
            if not pred:
                return {"error": "Prediction not found", "status": "not_found"}
            
            predictions = [pred]
        elif user_id:
            predictions = Predictions.query.filter_by(user_id=uuid.UUID(user_id)).order_by(Predictions.pred_date.desc()).all()
            if not predictions:
                return {"error": "No predictions found for this user", "status": "not_found"}
        else:
            return {"error": "Either prediction_id or user_id must be provided", "status": "bad_request"}
        
        result = []
        
        for pred in predictions:
            # Data dasar prediction
            pred_data = {
                "prediction_id": str(pred.pred_id),
                "user_id": str(pred.user_id) if pred.user_id else None,
                "guest_id": str(pred.guest_id) if pred.guest_id else None,
                "prediction_date": pred.pred_date.isoformat() if pred.pred_date else None,
                "input_data": {
                    "screen_time_hours": pred.screen_time,
                    "work_screen_hours": pred.work_screen,
                    "leisure_screen_hours": pred.leisure_screen,
                    "sleep_hours": pred.sleep_hours,
                    "sleep_quality_1_5": pred.sleep_quality,
                    "stress_level_0_10": pred.stress_level,
                    "productivity_0_100": pred.productivity,
                    "exercise_minutes_per_week": pred.exercise,
                    "social_hours_per_week": pred.social
                },
                "prediction_score": pred.pred_score,
                "details": []
            }
            
            details = PredDetails.query.filter_by(pred_id=pred.pred_id).all()
            for detail in details:
                detail_data = {
                    "factor_name": detail.factor_name,
                    "impact_score": detail.impact_score,
                    "advices": [],
                    "references": []
                }
                
                advices = Advices.query.filter_by(detail_id=detail.detail_id).all()
                for advice in advices:
                    detail_data["advices"].append(advice.advice_text)
                
                refs = References.query.filter_by(detail_id=detail.detail_id).all()
                for ref in refs:
                    detail_data["references"].append(ref.reference_link)
                
                pred_data["details"].append(detail_data)
            
            result.append(pred_data)
        
        if prediction_id:
            return {
                "status": "success",
                "data": result[0] if result else None
            }
        else:
            return {
                "status": "success",
                "data": result,
                "total_predictions": len(result)
            }
            
    except Exception as e:
        print(f"Error reading from PostgreSQL: {e}")
        return {
            "error": str(e),
            "status": "error"
        }

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
        
        # Generate unique prediction_id
        prediction_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # Initialize status sebagai processing
        try:
            store_prediction(prediction_id, {
                "status": "processing",
                "result": None,
                "created_at": created_at,
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
            args=(prediction_id, json_input, created_at, app)
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
    # Check Valkey for prediction status
    prediction_data = None
    try:
        prediction_data = fetch_prediction(prediction_id)
    except Exception as e:
        print(f"⚠️ Cache fetch warning: {e}")
        prediction_data = None
    
    if prediction_data:
        status = prediction_data["status"]
        
        if status == "processing":
            return jsonify({
                "status": "processing",
                "message": "Prediction is still being processed. Please try again in a moment."
            }), 202
        
        elif status == "partial":
            return jsonify({
                "status": "partial",
                "result": prediction_data["result"],
                "message": "Prediction ready. AI advice still processing.",
                "created_at": prediction_data["created_at"]
            }), 200
        
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
        
        # Check database (fallback) if not found in cache
        db_result = read_from_db(prediction_id=prediction_id)

        if db_result.get("status") == "success":
            data = db_result["data"]

            wellness_analysis = {
                "areas_for_improvement": [],
                "strengths": []
            }
            ai_advice = {}

            for detail in data.get("details", []):
                wellness_analysis["areas_for_improvement"].append({
                    "feature": detail["factor_name"],
                    "impact_score": detail["impact_score"]
                })
                ai_advice[detail["factor_name"]] = {
                    "advices": detail["advices"],
                    "references": detail["references"]
                }

            return jsonify({
                "status": "ready",
                "source": "database", 
                "created_at": data["prediction_date"],
                "completed_at": data["prediction_date"],
                "result": {
                    "prediction_score": data["prediction_score"],
                    "health_level": categorize_mental_health_score(data["prediction_score"]),
                    "wellness_analysis": wellness_analysis,
                    "advice": {
                        "description": "Historical result retrieved from database.",
                        "factors": ai_advice
                    }
                }
            }), 200
        
    return jsonify({
        "status": "not_found",
        "message": "Prediction ID not found"
    }), 404

@app.route('/advice', methods=['POST'])
def advice():
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