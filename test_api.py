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
    # "sleep_hours": 6.63,
    "sleep_hours": 4.5,
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
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n✅ PREDIKSI SUKSES!")
        print(f"Skor: {data['prediction'][0]:.2f}")

        print("="*60)
        
        if 'wellness_analysis' in data:
            print("📊 Wellness Analysis:")
            print(json.dumps(data['wellness_analysis'], indent=2))
        else:
            print("⚠️ Field 'wellness_analysis' tidak ditemukan dalam response.")
        
        if 'ai_advice' in data:
            print("="*60)
            print("\n🤖 AI Advice:")
            print(data['ai_advice'])
            
        print("="*60)
        
    else:
        print(f"\n❌ Error Status Code: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ Gagal connect. Pastikan 'python app.py' sudah jalan di terminal lain!")