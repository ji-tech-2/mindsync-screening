"""
AI Advice generation using Google Gemini
"""
import json
from google import genai
from google.genai import types

def get_ai_advice(prediction_score, category, wellness_analysis_result, api_key):
    """Generate personalized AI advice using Gemini."""
    
    # Extract top factors
    top_factors_list = []
    if wellness_analysis_result and 'areas_for_improvement' in wellness_analysis_result:
        top_factors_list = [
            item['feature'] 
            for item in wellness_analysis_result['areas_for_improvement'][:3]
        ]
    
    if top_factors_list:
        factors_inline_str = ", ".join(top_factors_list)
        factors_bullet_list = "\n".join([f"- {f}" for f in top_factors_list])
    else:
        factors_inline_str = "General Wellness"
        factors_bullet_list = "- General Wellness"

    # Trusted resources
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

    # Prompt for Gemini
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
        client = genai.Client(api_key=api_key)
        
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
        print(f"Gemini API Error: {e}")
        return {
            "description": "We encountered a temporary issue generating your personalized plan.",
            "factors": {}
        }
