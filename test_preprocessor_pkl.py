"""
Test script for preprocessor.pkl
Tests preprocessing on:
1. Healthy cluster average data
2. User dummy data
"""
import pickle
import pandas as pd
import numpy as np
import os

# --- REQUIRED IMPORTS FOR PICKLE RECONSTRUCTION ---
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel


# --- DEFINE CUSTOM CLEANER FUNCTION (must match notebook) ---
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


PREPROCESSOR_PATH = os.path.join('artifacts', 'preprocessor.pkl')
COEFFICIENTS_PATH = os.path.join('artifacts', 'model_coefficients.csv')
HEALTHY_CLUSTER_PATH = os.path.join('artifacts', 'healthy_cluster_avg.csv')


def load_preprocessor():
    """Load the preprocessor from pickle file"""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
    
    with open(PREPROCESSOR_PATH, 'rb') as file:
        preprocessor = pickle.load(file)
    return preprocessor


def create_healthy_cluster_input():
    """
    Create input data from healthy cluster averages.
    Based on healthy_cluster_avg.csv values.
    Uses the same format as the frontend sends to the API.
    """
    return {
        "age": 30,  # 29.94 rounded
        "gender": "Male",  # Dummy value (not in healthy_cluster_avg)
        "occupation": "Working Professional",  # Dummy value
        "work_mode": "Hybrid",  # Dummy value
        "work_screen_hours": 1.18,
        "leisure_screen_hours": 4.75,
        "sleep_hours": 8.05,
        "sleep_quality_1_5": 3,  # 2.51 rounded
        "stress_level_0_10": 4.29,
        "productivity_0_100": 82.01,
        "exercise_minutes_per_week": 114,  # 114.12 rounded
        "social_hours_per_week": 10.18,
        "mental_wellness_index_0_100": None  # Target variable, set to None
    }


def create_user_dummy_input():
    """
    Create a dummy user input for testing.
    Simulates a typical user with moderate screen time habits.
    """
    return {
        "age": 25,
        "gender": "Female",
        "occupation": "Student",
        "work_mode": "Remote",
        "work_screen_hours": 6.5,
        "leisure_screen_hours": 3.0,
        "sleep_hours": 7.0,
        "sleep_quality_1_5": 3,
        "stress_level_0_10": 5.5,
        "productivity_0_100": 70.0,
        "exercise_minutes_per_week": 90,
        "social_hours_per_week": 8.0,
        "mental_wellness_index_0_100": None
    }


def create_unhealthy_user_input():
    """
    Create an unhealthy user input for comparison.
    High screen time, poor sleep, high stress.
    """
    return {
        "age": 35,
        "gender": "Male",
        "occupation": "Working Professional",
        "work_mode": "On-site",
        "work_screen_hours": 10.0,
        "leisure_screen_hours": 5.0,
        "sleep_hours": 5.0,
        "sleep_quality_1_5": 1,
        "stress_level_0_10": 8.5,
        "productivity_0_100": 45.0,
        "exercise_minutes_per_week": 30,
        "social_hours_per_week": 2.0,
        "mental_wellness_index_0_100": None
    }


def prepare_dataframe(input_data):
    """
    Convert input dict to DataFrame in the expected format.
    Removes mental_wellness_index_0_100 as it's the target variable.
    """
    # Create a copy to avoid modifying original
    data = input_data.copy()
    
    # Remove target variable (if present and None)
    if 'mental_wellness_index_0_100' in data:
        del data['mental_wellness_index_0_100']
    
    # Create DataFrame
    df = pd.DataFrame([data])
    return df


def test_preprocessor():
    """Main test function"""
    print("=" * 60)
    print("PREPROCESSOR TEST SCRIPT")
    print("=" * 60)
    
    # Load preprocessor
    try:
        preprocessor = load_preprocessor()
        print(f"\n✓ Successfully loaded preprocessor from {PREPROCESSOR_PATH}")
        print(f"  Preprocessor type: {type(preprocessor)}")
    except Exception as e:
        print(f"\n✗ Failed to load preprocessor: {e}")
        return
    
    # Test cases
    test_cases = [
        ("Healthy Cluster Average", create_healthy_cluster_input()),
        ("User Dummy (Moderate)", create_user_dummy_input()),
        ("Unhealthy User", create_unhealthy_user_input()),
    ]
    
    for name, input_data in test_cases:
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print("=" * 60)
        
        # Show input
        print("\n📥 Input Data:")
        for key, value in input_data.items():
            print(f"   {key}: {value}")
        
        # Prepare DataFrame
        df = prepare_dataframe(input_data)
        print(f"\n📊 DataFrame shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Run preprocessor
        try:
            transformed = preprocessor.transform(df)
            print(f"\n✓ Preprocessing successful!")
            print(f"   Output shape: {transformed.shape}")
            print(f"   Output type: {type(transformed)}")
            
            # Show transformed data
            if hasattr(transformed, 'toarray'):
                # Sparse matrix
                transformed_arr = transformed.toarray()
            else:
                transformed_arr = transformed
            
            print(f"\n📤 Transformed features ({transformed_arr.shape[1]} features):")
            print(f"   Values: {transformed_arr[0][:10]}...")  # First 10 features
            print(f"   Min: {transformed_arr.min():.4f}, Max: {transformed_arr.max():.4f}")
            print(f"   Mean: {transformed_arr.mean():.4f}, Std: {transformed_arr.std():.4f}")
            
        except Exception as e:
            print(f"\n✗ Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print("=" * 60)


def load_healthy_cluster_from_csv():
    """Load healthy cluster data from CSV file"""
    if not os.path.exists(HEALTHY_CLUSTER_PATH):
        raise FileNotFoundError(f"Healthy cluster file not found at {HEALTHY_CLUSTER_PATH}")
    
    df = pd.read_csv(HEALTHY_CLUSTER_PATH)
    return df


def load_coefficients():
    """Load model coefficients from CSV"""
    if not os.path.exists(COEFFICIENTS_PATH):
        raise FileNotFoundError(f"Coefficients file not found at {COEFFICIENTS_PATH}")
    
    df = pd.read_csv(COEFFICIENTS_PATH)
    return df


def analyze_wellness_factors():
    """
    Analyze wellness factors by comparing healthy cluster vs user input.
    Formula: (preprocessed_healthy - preprocessed_user) * coefficient
    """
    print("=" * 80)
    print("WELLNESS FACTOR ANALYSIS")
    print("=" * 80)
    
    # Load components
    try:
        preprocessor = load_preprocessor()
        print(f"\n✓ Loaded preprocessor")
        
        coefficients_df = load_coefficients()
        print(f"✓ Loaded coefficients ({len(coefficients_df)} features)")
        
        # Load healthy cluster from CSV
        healthy_df = load_healthy_cluster_from_csv()
        print(f"✓ Loaded healthy cluster from CSV")
        
    except Exception as e:
        print(f"\n✗ Failed to load components: {e}")
        return
    
    # Create user dummy input
    user_input = create_user_dummy_input()
    user_df = prepare_dataframe(user_input)
    
    print(f"\n{'=' * 80}")
    print("INPUT COMPARISON")
    print("=" * 80)
    print("\n📊 Healthy Cluster (from CSV):")
    print(healthy_df.to_string(index=False))
    print(f"\n👤 User Input:")
    for key, value in user_input.items():
        if key != 'mental_wellness_index_0_100':
            print(f"   {key}: {value}")
    
    # Preprocess both
    try:
        healthy_preprocessed = preprocessor.transform(healthy_df)
        user_preprocessed = preprocessor.transform(user_df)
        
        # Convert to arrays if sparse
        if hasattr(healthy_preprocessed, 'toarray'):
            healthy_arr = healthy_preprocessed.toarray()[0]
            user_arr = user_preprocessed.toarray()[0]
        else:
            healthy_arr = healthy_preprocessed[0]
            user_arr = user_preprocessed[0]
        
        print(f"\n✓ Preprocessing successful")
        print(f"   Healthy preprocessed shape: {healthy_arr.shape}")
        print(f"   User preprocessed shape: {user_arr.shape}")
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate differences and multiply by coefficients
    print(f"\n{'=' * 80}")
    print("WELLNESS GAP ANALYSIS")
    print("=" * 80)
    print("\nFormula: (Healthy - User) × Coefficient = Impact Score")
    print("Positive Impact: User is lacking this factor (needs improvement)")
    print("Negative Impact: User has too much of this negative factor")
    
    results = []
    
    for idx, row in coefficients_df.iterrows():
        feature_name = row['Feature']
        coefficient = row['Coefficient']
        
        # Find matching feature index (assumes order matches)
        if idx < len(healthy_arr):
            healthy_value = healthy_arr[idx]
            user_value = user_arr[idx]
            
            # Calculate difference and impact
            difference = healthy_value - user_value
            impact_score = difference * coefficient
            
            results.append({
                'Feature': feature_name,
                'Coefficient': coefficient,
                'Healthy_Value': healthy_value,
                'User_Value': user_value,
                'Difference': difference,
                'Impact_Score': impact_score
            })
    
    results_df = pd.DataFrame(results)
    
    # Separate positive and negative factors
    positive_factors = results_df[results_df['Impact_Score'] > 0].sort_values('Impact_Score', ascending=False)
    negative_factors = results_df[results_df['Impact_Score'] < 0].sort_values('Impact_Score', ascending=True)
    
    # Display positive factors (lacking/needs improvement)
    print(f"\n{'🟢 POSITIVE FACTORS - USER IS LACKING (Top 10)':.^80}")
    print("These are areas where improvement would boost wellness score:\n")
    
    if len(positive_factors) > 0:
        for idx, row in positive_factors.head(10).iterrows():
            print(f"  {row['Feature']}")
            print(f"    ├─ Impact Score: +{row['Impact_Score']:.4f}")
            print(f"    ├─ Coefficient: {row['Coefficient']:.4f}")
            print(f"    ├─ Healthy: {row['Healthy_Value']:.4f} | User: {row['User_Value']:.4f}")
            print(f"    └─ Gap: {row['Difference']:.4f} (User needs MORE)\n")
    else:
        print("  No positive factors found.\n")
    
    # Display negative factors (too much/needs reduction)
    print(f"\n{'🔴 NEGATIVE FACTORS - USER HAS TOO MUCH (Top 10)':.^80}")
    print("These are areas where reduction would boost wellness score:\n")
    
    if len(negative_factors) > 0:
        for idx, row in negative_factors.head(10).iterrows():
            print(f"  {row['Feature']}")
            print(f"    ├─ Impact Score: {row['Impact_Score']:.4f}")
            print(f"    ├─ Coefficient: {row['Coefficient']:.4f}")
            print(f"    ├─ Healthy: {row['Healthy_Value']:.4f} | User: {row['User_Value']:.4f}")
            print(f"    └─ Gap: {row['Difference']:.4f} (User needs LESS)\n")
    else:
        print("  No negative factors found.\n")
    
    # Summary statistics
    print(f"\n{'📊 SUMMARY STATISTICS':.^80}")
    total_positive_impact = positive_factors['Impact_Score'].sum()
    total_negative_impact = negative_factors['Impact_Score'].sum()
    net_impact = results_df['Impact_Score'].sum()
    
    print(f"\n  Total Positive Impact:  +{total_positive_impact:.4f} (areas lacking)")
    print(f"  Total Negative Impact:  {total_negative_impact:.4f} (areas in excess)")
    print(f"  Net Impact Score:       {net_impact:.4f}")
    print(f"\n  Number of Positive Factors: {len(positive_factors)}")
    print(f"  Number of Negative Factors: {len(negative_factors)}")
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Run both tests
    test_preprocessor()
    print("\n\n")
    analyze_wellness_factors()
