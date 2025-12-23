import requests
import json

# API Endpoint
url = "http://localhost:5000/predict"



sample_data = {
    "age": 33,
    "gender": "Female",
    "occupation": "Employed",
    "work_mode": "Remote",
    # "screen_time_hours": 10.79,
    "work_screen_hours": 5.44,
    "leisure_screen_hours": 5.35,
    "sleep_hours": 6.63,
    "sleep_quality_1_5": 1,
    "stress_level_0_10": 9.3,
    "productivity_0_100": 44.7,
    "exercise_minutes_per_week": 127,
    "social_hours_per_week": 0.7
}

print(f"Sending request to {url}...")
print("Input Data:")
print(json.dumps(sample_data, indent=2))

try:
    response = requests.post(
        url, 
        headers={"Content-Type": "application/json"},
        data=json.dumps(sample_data)
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print("API Response:")
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.ConnectionError:
    print("\n[Error] Could not connect. Is 'python app.py' running?")