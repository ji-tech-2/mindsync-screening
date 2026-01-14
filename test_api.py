import requests
import json

# API Endpoints
predict_url = "http://localhost:5000/predict"
advice_url = "http://localhost:5000/advice"

sample_data = {
    "age": 33,
    "gender": "Female",
    "occupation": "Employed",
    "work_mode": "Remote",
    "work_screen_hours": 5.44,
    "leisure_screen_hours": 5.35,
    "sleep_hours": 4.5,
    "sleep_quality_1_5": 1,
    "stress_level_0_10": 9.3,
    "productivity_0_100": 44.7,
    "exercise_minutes_per_week": 127,
    "social_hours_per_week": 0.7
}

print(f"\n--- STEP 1: PREDICT ---")
try:
    # 1. Request Predict
    res_predict = requests.post(predict_url, json=sample_data)
    
    if res_predict.status_code == 200:
        data_predict = res_predict.json()
        print("✅ Predict Sukses!")
        print(f"Score: {data_predict['prediction'][0]}")
        print(f"Category: {data_predict['mental_health_category']}")
        
        # 2. Request Advice (Pakai data output dari Predict)
        print(f"\n--- STEP 2: GET ADVICE ---")
        
        # Susun payload persis seperti output /predict
        advice_payload = {
            "prediction": data_predict['prediction'], # Bisa kirim list [35.5]
            "mental_health_category": data_predict['mental_health_category'],
            "wellness_analysis": data_predict['wellness_analysis']
        }
        
        res_advice = requests.post(advice_url, json=advice_payload)
        
        if res_advice.status_code == 200:
            data_advice = res_advice.json()
            print("✅ Advice Sukses!")
            print(json.dumps(data_advice['ai_advice'], indent=2))
        else:
            print(f"❌ Advice Error: {res_advice.text}")
            
    else:
        print(f"❌ Predict Error: {res_predict.text}")

except Exception as e:
    print(f"Connection Error: {e}")