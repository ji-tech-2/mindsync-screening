"""
Test script for Flask API (app.py)
Tests the /predict endpoint with dummy user inputs
"""
import requests
import json

# API Configuration
API_URL = "http://localhost:5000/predict"

def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)

def test_predict_endpoint(test_name, user_input):
    """
    Test the /predict endpoint with user input
    
    Args:
        test_name: Name of the test case
        user_input: Dictionary with user data
    """
    print_separator()
    print(f"TEST: {test_name}")
    print_separator()
    
    # Show input
    print("\n📥 RAW INPUT:")
    for key, value in user_input.items():
        print(f"   {key}: {value}")
    
    try:
        # Send POST request
        response = requests.post(API_URL, json=user_input, timeout=10)
        
        # Check response status
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Request successful (Status: {response.status_code})")
            
            # Show prediction
            if "prediction" in result:
                print(f"\n🎯 PREDICTION:")
                print(f"   Mental Wellness Score: {result['prediction'][0]:.2f}/100")
            
            # Show top contributor factors
            if "top_contributor_factors" in result:
                factors = result["top_contributor_factors"]
                
                # Positive factors (user is lacking)
                print(f"\n🟢 POSITIVE FACTORS - USER IS LACKING (Top 5):")
                if factors.get("positive_factors"):
                    for i, factor in enumerate(factors["positive_factors"], 1):
                        print(f"\n   {i}. {factor['feature']}")
                        print(f"      ├─ Impact Score: +{factor['impact_score']:.4f}")
                        print(f"      ├─ Coefficient: {factor['coefficient']:.4f}")
                        print(f"      ├─ Healthy: {factor['healthy_value']:.4f} | User: {factor['user_value']:.4f}")
                        print(f"      └─ Gap: {factor['gap']:.4f} (User needs MORE)")
                else:
                    print("   No positive factors found (user is doing well!)")
                
                # Negative factors (user has too much)
                print(f"\n🔴 NEGATIVE FACTORS - USER HAS TOO MUCH (Top 5):")
                if factors.get("negative_factors"):
                    for i, factor in enumerate(factors["negative_factors"], 1):
                        print(f"\n   {i}. {factor['feature']}")
                        print(f"      ├─ Impact Score: {factor['impact_score']:.4f}")
                        print(f"      ├─ Coefficient: {factor['coefficient']:.4f}")
                        print(f"      ├─ Healthy: {factor['healthy_value']:.4f} | User: {factor['user_value']:.4f}")
                        print(f"      └─ Gap: {factor['gap']:.4f} (User needs LESS)")
                else:
                    print("   No negative factors found (user is doing well!)")
            
            # Show raw response
            print(f"\n📤 RAW JSON RESPONSE:")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"\n✗ Request failed (Status: {response.status_code})")
            print(f"   Error: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Connection failed!")
        print(f"   Make sure the Flask app is running on {API_URL}")
        print(f"   Run: python app.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print()

def create_healthy_user():
    """Create a healthy user profile"""
    return {
        "age": 30,
        "gender": "Male",
        "occupation": "Working Professional",
        "work_mode": "Hybrid",
        "work_screen_hours": 1.18,
        "leisure_screen_hours": 4.75,
        "sleep_hours": 8.05,
        "sleep_quality_1_5": 3,
        "stress_level_0_10": 4.29,
        "productivity_0_100": 82.01,
        "exercise_minutes_per_week": 114,
        "social_hours_per_week": 10.18
    }

def create_moderate_user():
    """Create a moderate user profile"""
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
        "social_hours_per_week": 8.0
    }

def create_unhealthy_user():
    """Create an unhealthy user profile"""
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
        "social_hours_per_week": 2.0
    }

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✓ API is running")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ API responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ API is not running!")
        print("  Please start the Flask app first: python app.py")
        return False
    except Exception as e:
        print(f"✗ Error checking API: {e}")
        return False

def main():
    """Main test function"""
    print_separator("=", 80)
    print("FLASK API TEST SCRIPT")
    print_separator("=", 80)
    
    # Check if API is running
    print("\n🔍 Checking API status...")
    if not check_api_status():
        print("\n⚠️  Cannot proceed with tests. Start the Flask app and try again.")
        return
    
    print()
    
    # Test cases
    test_cases = [
        ("Healthy User (Baseline)", create_healthy_user()),
        ("Moderate User (Average Screen Time)", create_moderate_user()),
        ("Unhealthy User (High Stress, Poor Sleep)", create_unhealthy_user()),
    ]
    
    # Run tests
    for test_name, user_input in test_cases:
        test_predict_endpoint(test_name, user_input)
        print()
    
    print_separator("=", 80)
    print("ALL TESTS COMPLETE")
    print_separator("=", 80)

if __name__ == "__main__":
    main()
