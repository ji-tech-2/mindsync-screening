"""
AI Advice generation using Google Gemini
"""

import json
import logging
import threading
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
MODEL_NAME = "gemini-2.5-flash"
_rotation_index = 0
_rotation_lock = threading.Lock()

# Trusted resources for AI advice
TRUSTED_SOURCES = """  # noqa: E501
    - https://www.mayoclinichealthsystem.org/hometown-health/speaking-of-health/5-ways-to-get-better-sleep  # noqa: E501
    - https://www.aoa.org/healthy-eyes/eye-and-vision-conditions/computer-vision-syndrome  # noqa: E501
    - https://www.sleepfoundation.org/sleep-hygiene
    - https://www.cdc.gov/sleep/about/index.html
    - https://www.apa.org/topics/stress/tips
    - https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health
    - https://www.who.int/news-room/fact-sheets/detail/physical-activity
    - https://www.nia.nih.gov/health/brain-health/cognitive-health-and-older-adults  # noqa: E501
    - https://www.health.harvard.edu/staying-healthy/the-health-benefits-of-strong-relationships  # noqa: E501
    - https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-minute-boost-your-health-and-productivity-with-activity-snacks/  # noqa: E501
    - https://youtu.be/dlgCJd1cfy8?si=mmk2X8vvUGjtvWBJ
    """

# AI Prompt Template - separated from code for maintainability
ADVICE_PROMPT_TEMPLATE = """
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
            "references": [
                {{"title": "Resource Title 1", "url": "URL 1"}},
                {{"title": "Resource Title 2", "url": "URL 2"}}
            ]
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
     b. "references" (Array of Objects): Select 1 to 3 relevant resources for this factor from the list below.
        Each reference must be a JSON object with "title" (a descriptive name for the resource) and "url" (the link).
        Sources:
        {trusted_sources}
        (If no link matches perfectly, use a general mental health link).

Tone: Professional, warm, non-judgmental.
Language: English (Standard US).
"""


def gemini_rotation(prompt, api_keys_string):
    """Helper function for API key queue (round robin with fallback)."""
    global _rotation_index

    if not api_keys_string:
        logger.error("No API Keys provided in config.")
        return None

    if isinstance(api_keys_string, (list, tuple)):
        keys_pool = [str(k).strip() for k in api_keys_string if str(k).strip()]
    else:
        keys_pool = [k.strip() for k in str(api_keys_string).split(",") if k.strip()]

    num_keys = len(keys_pool)

    if num_keys == 0:
        logger.error("API Key list is empty after parsing.")
        return None

    for _ in range(num_keys):
        with _rotation_lock:
            current_idx = _rotation_index % num_keys
            current_key = keys_pool[current_idx]
            _rotation_index += 1

        key_id = current_key[-4:]

        try:
            logger.info(
                "[Queue] Using Key ID: ...%s (Queue index %s)",
                key_id,
                current_idx + 1,
            )

            client = genai.Client(api_key=current_key)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7, response_mime_type="application/json"
                ),
            )

            result = json.loads(response.text.strip())
            logger.info("Gemini request success with Key ...%s", key_id)
            return result

        except Exception as e:
            logger.warning("Gemini request failed with Key ...%s: %s", key_id, e)
            logger.info("Trying next key in queue...")
            continue

    logger.error("All API keys in queue failed.")
    return None


def get_ai_advice(prediction_score, category, wellness_analysis_result, api_keys_pool):
    """Generate personalized AI advice using Gemini."""
    logger.info(
        f"Generating AI advice for category: {category}, score: {prediction_score:.2f}"
    )

    # Extract top factors
    top_factors_list = []
    if wellness_analysis_result and "areas_for_improvement" in wellness_analysis_result:
        top_factors_list = [
            item["feature"]
            for item in wellness_analysis_result["areas_for_improvement"][:3]
        ]

    logger.debug("Top improvement factors: %s", top_factors_list)

    if top_factors_list:
        factors_inline_str = ", ".join(top_factors_list)
        factors_bullet_list = "\n".join([f"- {f}" for f in top_factors_list])
    else:
        factors_inline_str = "General Wellness"
        factors_bullet_list = "- General Wellness"

    # Build prompt from template using .format() for security and maintainability
    prompt = ADVICE_PROMPT_TEMPLATE.format(
        category=category,
        prediction_score=prediction_score,
        factors_bullet_list=factors_bullet_list,
        factors_inline_str=factors_inline_str,
        trusted_sources=TRUSTED_SOURCES,
    )

    try:
        result = gemini_rotation(prompt, api_keys_pool)
        if result is not None:
            return result

        return {
            "description": (
                "We encountered a temporary issue generating your personalized plan."
            ),
            "factors": {},
        }
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            "description": (
                "We encountered a temporary issue generating your personalized plan."
            ),
            "factors": {},
        }


def get_weekly_advice(top_factors, api_key):
    """
    Generate AI advice for the top critical factors from the past week.

    Args:
        top_factors: List of dicts with 'factor_name', 'count', and 'avg_impact_score'
        api_key: Gemini API key

    Returns:
        Dict with description and advice for each factor
    """
    if not top_factors:
        return {
            "description": "Great job! No critical factors detected this week.",
            "factors": {},
        }

    # Build factor context
    factors_list = [f["factor_name"] for f in top_factors]
    factors_inline_str = ", ".join(factors_list)  # noqa: E501
    factors_bullet_list = "\n".join(  # noqa: E501
        [
            f"- {f['factor_name']} (appeared {f['count']} times, "
            f"avg impact: {f['avg_impact_score']:.2f})"
            for f in top_factors
        ]
    )

    prompt = f"""
    Role: You are 'MindSync', a mental health AI advisor providing a weekly wellness summary.

    Context:
    Based on the user's activity this past week, here are the most frequent areas that need attention:
    {factors_bullet_list}

    Task:
    Generate a weekly summary with advice. Return STRICTLY as a JSON object:

    {{
        "description": "String (warm weekly summary paragraph)",
        "factors": {{
            "FACTOR_NAME_1": {{
                "advices": ["String 1", "String 2", "String 3"],
                "references": [
                    {{"title": "Resource Title 1", "url": "URL 1"}},
                    {{"title": "Resource Title 2", "url": "URL 2"}}
                ]
            }},
            "FACTOR_NAME_2": {{
                ...
            }}
        }}
    }}

    Requirements:

    1. "description":
       - Write a warm, encouraging weekly summary.
       - Acknowledge that {factors_inline_str} have been recurring challenges this week.
       - Encourage progress and self-compassion.

    2. "factors":
       - Create a key for EACH of these critical factors: {factors_inline_str}.
       - For each factor, provide:
         a. "advices" (Array of Strings): Exactly 3 actionable weekly goals/tips specific to that factor.  # noqa: E501
         b. "references" (Array of Objects): 1-3 relevant resources from the list below.
            Each reference must be a JSON object with "title" (a descriptive name for the resource) and "url" (the link).  # noqa: E501
            Sources:
            {TRUSTED_SOURCES}

    Tone: Professional, warm, encouraging, focused on weekly improvement.
    Language: English (Standard US).
    """

    try:
        result = gemini_rotation(prompt, api_key)
        if result is not None:
            return result

        return {
            "description": (
                "We encountered a temporary issue generating your " "weekly summary."
            ),
            "factors": {},
        }
    except Exception as e:
        print(f"Gemini API Error (weekly advice): {e}")
        return {
            "description": (
                "We encountered a temporary issue generating your " "weekly summary."
            ),
            "factors": {},
        }


def get_daily_advice(top_factors, api_key):
    """
    Generate a short AI suggestion for today's areas of improvement.

    Args:
        top_factors: List of dicts with 'factor_name' and 'impact_score'
        api_key: Gemini API key

    Returns:
        each factor will now have a short daily suggestion string
    """
    if not top_factors:
        return "You're doing great today! Keep up the good work."

    # Build factor context
    factors_list = [f["factor_name"] for f in top_factors]
    factors_inline_str = ", ".join(factors_list)  # noqa: E501
    factors_bullet_list = "\n".join(  # noqa: E501
        [
            f"- {f['factor_name']} (impact score: {f['impact_score']:.2f})"
            for f in top_factors
        ]
    )

    prompt = f"""
    Role: You are 'MindSync', a mental health AI advisor providing a daily wellness tip.

    Context:
    Based on the user's check-in today, here are the areas that need attention:
    {factors_bullet_list}

    Task:
    Generate a short, actionable daily suggestion in one sentence for each of these factors.
    Return STRICTLY as a JSON object with the following structure:

    {{
        "FACTOR_NAME_1": "One sentence of actionable advice",
        "FACTOR_NAME_2": "One sentence of actionable advice"
    }}

    Requirements:
    - Create a key for EACH of these factors: {factors_inline_str}
    - Each value must be exactly one sentence
    - Be warm, encouraging, and actionable
    - Include one specific tip the user can do today for each factor  # noqa: E501

    Tone: Friendly, encouraging, actionable.
    Language: English (Standard US).
    """

    try:
        result = gemini_rotation(prompt, api_key)
        if result is not None:
            return result

        return {
            "error": (
                "Take a moment to focus on your wellness today. "
                "Small steps lead to big improvements!"
            )
        }
    except Exception as e:
        print(f"Gemini API Error (daily advice): {e}")
        return {
            "error": (
                "Take a moment to focus on your wellness today. "
                "Small steps lead to big improvements!"
            )
        }
