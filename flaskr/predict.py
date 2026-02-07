"""
Prediction routes and business logic
"""
import uuid
import threading
import pandas as pd
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy.orm import selectinload
from sqlalchemy import func

from . import model, cache, ai
from .db import db, Predictions, PredDetails, Advices, References, UserStreaks, is_valid_uuid

# Constants
FACTOR_TYPE_IMPROVEMENT = 'improvement'
FACTOR_TYPE_STRENGTH = 'strengths'

bp = Blueprint('predict', __name__, url_prefix='/')

# ===================== #
#   PREDICTION ROUTES   #
# ===================== #

@bp.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - returns prediction_id immediately."""
    if model.model is None:
        return jsonify({
            "error": "Model failed to load. Check server logs."
        }), 500
    
    # Check if at least one storage backend is available
    db_enabled = not current_app.config.get('DB_DISABLED', False)
    cache_available = cache.is_available()
    
    if not db_enabled and not cache_available:
        return jsonify({
            "error": "No storage backend available",
            "message": "Both database and cache are unavailable. Cannot process predictions."
        }), 503

    try:
        json_input = request.get_json()
        
        # Generate unique prediction_id
        prediction_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # Initialize status as processing
        cache.store_prediction(prediction_id, {
            "status": "processing",
            "result": None,
            "created_at": created_at,
        })
        
        # Start background processing
        thread = threading.Thread(
            target=process_prediction,
            args=(prediction_id, json_input, created_at, current_app._get_current_object())
        )
        thread.daemon = True
        thread.start()
        
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

@bp.route('/result/<prediction_id>', methods=['GET'])
def get_result(prediction_id):
    """Check prediction status and get results."""
    
    # Validate UUID
    if not is_valid_uuid(prediction_id):
        return jsonify({
            "error": "Invalid ID format",
            "message": "The provided ID must be a standard UUID."
        }), 400
    
    # Check cache first
    prediction_data = cache.fetch_prediction(prediction_id)
    
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
    
    # Fallback to database if enabled
    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "status": "not_found",
            "message": "Prediction ID not found (DB disabled)."
        }), 404

    db_result = read_from_db(prediction_id=prediction_id)

    if db_result.get("status") == "success":
        data = db_result["data"]
        return jsonify({
            "status": "ready",
            "source": "database",
            "created_at": data["prediction_date"],
            "completed_at": data["prediction_date"],
            "result": format_db_output(data)
        }), 200
    
    return jsonify(db_result), 404

@bp.route('/advice', methods=['POST'])
def advice():
    """Generate AI advice (for testing/manual calls)."""
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
            return jsonify({
                "error": "Missing inputs from /predict result",
                "status": "error"
            }), 400
        
        api_key = current_app.config.get('GEMINI_API_KEY')
        ai_advice = ai.get_ai_advice(prediction_score, category, analysis, api_key)
        
        return jsonify({
            "ai_advice": ai_advice,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    
@bp.route('/streak/<user_id>', methods=['GET'])
def get_streak(user_id):
    """Get current streak status (Daily & Weekly) for a user."""

    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "status": "error",
            "message": "Database is disabled. Streak data unavailable."
        }), 503

    if not is_valid_uuid(user_id):
        return jsonify({"error": "Invalid user_id format. Must be a valid UUID string."}), 400
        
    try:
        # Use existing database session
        streak_record = UserStreaks.query.get(uuid.UUID(user_id))
        
        if streak_record:
            return jsonify({
                "status": "success",
                "data": streak_record.to_dict()
            }), 200
        else:
            # If no record exists, return default 0 values (User hasn't started yet)
            return jsonify({
                "status": "success",
                "data": {
                    "user_id": user_id,
                    "daily": {"current": 0, "last_date": None},
                    "weekly": {"current": 0, "last_date": None}
                }
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
    
@bp.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """ Get full history of predictions for a user."""
    
    # Check DB Status
    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "status": "error", 
            "message": "Database is disabled. History unavailable."
        }), 503

    # Validate UUID
    if not is_valid_uuid(user_id):
        return jsonify({"error": "Invalid User ID format"}), 400
        
    try:
        db_result = read_from_db(user_id=user_id)
        if db_result.get("status") == "success":
            # Format list of results
            formatted_history = []
            for item in db_result["data"]:
                formatted_data = format_db_output(item)
                formatted_item = {
                    "prediction_id": item["prediction_id"],
                    "prediction_score": item["prediction_score"],
                    "health_level": model.categorize_mental_health_score(item["prediction_score"]),
                    "created_at": item["prediction_date"],
                    "wellness_analysis": formatted_data["wellness_analysis"],
                    "advice": formatted_data["advice"]
                }
                formatted_history.append(formatted_item)
            
            return jsonify({
                "status": "success",
                "count": len(formatted_history),
                "data": formatted_history
            }), 200
        
        elif db_result.get("status") == "not_found":
            # User has no history yet, return empty list (not error)
            return jsonify({
                "status": "success",
                "count": 0,
                "data": []
            }), 200
            
        else:
            return jsonify(db_result), 400
            
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@bp.route('/weekly-critical-factors', methods=['GET'])
def get_weekly_critical_factors():
    """
    Get the top 3 most frequent areas_for_improvement from the last week.
    Also generates AI advice for these critical factors.
    
    Query params:
        user_id (optional): Filter by specific user
        days (optional): Number of days to look back (default: 7)
    """
    from flask import current_app
    
    # Check if database is enabled
    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "error": "Database is disabled",
            "message": "This endpoint requires database access.",
            "status": "unavailable"
        }), 503
    
    try:
        # Parse query parameters
        user_id = request.args.get('user_id')
        days = request.args.get('days', 7, type=int)
        
        # Validate user_id if provided
        if user_id and not is_valid_uuid(user_id):
            return jsonify({
                "error": "Invalid user_id format",
                "message": "user_id must be a valid UUID.",
                "status": "bad_request"
            }), 400
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Build query to get top critical factors
        query = db.session.query(
            PredDetails.factor_name,
            func.count(PredDetails.factor_name).label('occurrence_count'),
            func.avg(PredDetails.impact_score).label('avg_impact_score')
        ).join(
            Predictions, Predictions.pred_id == PredDetails.pred_id
        ).filter(
            Predictions.pred_date >= start_date,
            Predictions.pred_date <= end_date,
            PredDetails.impact_score > 0  # Only areas for improvement (positive impact)
        )
        
        # Filter by user if provided
        if user_id:
            query = query.filter(Predictions.user_id == uuid.UUID(user_id))
        
        # Group by factor and order by frequency
        results = query.group_by(
            PredDetails.factor_name
        ).order_by(
            func.count(PredDetails.factor_name).desc()
        ).limit(3).all()
        
        # Format the results
        top_factors = []
        for row in results:
            top_factors.append({
                "factor_name": row.factor_name,
                "count": row.occurrence_count,
                "avg_impact_score": float(row.avg_impact_score) if row.avg_impact_score else 0.0
            })
        
        # Get additional stats
        total_predictions = db.session.query(
            func.count(Predictions.pred_id)
        ).filter(
            Predictions.pred_date >= start_date,
            Predictions.pred_date <= end_date
        )
        
        if user_id:
            total_predictions = total_predictions.filter(
                Predictions.user_id == uuid.UUID(user_id)
            )
        
        total_count = total_predictions.scalar() or 0
        
        # Generate AI advice for these critical factors
        api_key = current_app.config.get('GEMINI_API_KEY')
        ai_advice = None
        
        if top_factors and api_key:
            ai_advice = ai.get_weekly_advice(top_factors, api_key)
        elif not api_key:
            ai_advice = {
                "description": "AI advice unavailable (API key not configured)",
                "factors": {}
            }
        elif not top_factors:
            ai_advice = {
                "description": "No wellness data recorded in this period. Complete wellness checks to get personalized weekly insights!",
                "factors": {}
            }
        
        return jsonify({
            "status": "success",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "stats": {
                "total_predictions": total_count,
                "user_id": user_id
            },
            "top_critical_factors": top_factors,
            "advice": ai_advice
        }), 200
    
    except Exception as e:
        print(f"Error in weekly critical factors: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@bp.route('/daily-suggestion', methods=['GET'])
def get_daily_suggestion():
    """
    Get AI-powered daily suggestion based on today's areas of improvement.
    
    Query params:
        user_id (required): User UUID to get today's suggestions for
    """
    from flask import current_app
    
    # Check if database is enabled
    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "error": "Database is disabled",
            "message": "This endpoint requires database access.",
            "status": "unavailable"
        }), 503
    
    try:
        # Parse query parameters
        user_id = request.args.get('user_id')
        
        # Validate user_id
        if not user_id:
            return jsonify({
                "error": "Missing user_id",
                "message": "user_id query parameter is required.",
                "status": "bad_request"
            }), 400
        
        if not is_valid_uuid(user_id):
            return jsonify({
                "error": "Invalid user_id format",
                "message": "user_id must be a valid UUID.",
                "status": "bad_request"
            }), 400
        
        # Calculate today's date range (midnight to midnight)
        today = datetime.now().date()
        start_of_day = datetime.combine(today, datetime.min.time())
        end_of_day = datetime.combine(today, datetime.max.time())
        
        # Query today's predictions for this user
        results = db.session.query(
            PredDetails.factor_name,
            func.avg(PredDetails.impact_score).label('avg_impact_score')
        ).join(
            Predictions, Predictions.pred_id == PredDetails.pred_id
        ).filter(
            Predictions.user_id == uuid.UUID(user_id),
            Predictions.pred_date >= start_of_day,
            Predictions.pred_date <= end_of_day,
            PredDetails.impact_score > 0  # Only areas for improvement
        ).group_by(
            PredDetails.factor_name
        ).order_by(
            func.avg(PredDetails.impact_score).desc()
        ).limit(3).all()
        
        # Format the results
        top_factors = []
        for row in results:
            top_factors.append({
                "factor_name": row.factor_name,
                "impact_score": float(row.avg_impact_score) if row.avg_impact_score else 0.0
            })
        
        # Get today's prediction count
        prediction_count = db.session.query(
            func.count(Predictions.pred_id)
        ).filter(
            Predictions.user_id == uuid.UUID(user_id),
            Predictions.pred_date >= start_of_day,
            Predictions.pred_date <= end_of_day
        ).scalar() or 0
        
        # Generate AI advice for today's factors
        api_key = current_app.config.get('GEMINI_API_KEY')
        ai_advice = None
        
        if top_factors and api_key:
            ai_advice = ai.get_daily_advice(top_factors, api_key)
        elif not api_key:
            ai_advice = {"message": "AI advice unavailable. Take a moment to reflect on your wellness today."}
        elif not top_factors:
            ai_advice = {"message": "No check-ins yet today. Complete a wellness check to get personalized suggestions!"}
        
        # Keep these variables for future database storage
        response_data = {
            "date": today.isoformat(),
            "user_id": user_id,
            "stats": {
                "predictions_today": prediction_count
            },
            "areas_of_improvement": top_factors,
        }
        # TODO: Save response_data to database when schema is updated
        
        return jsonify({
            "status": "success",
            "suggestion": ai_advice
        }), 200
    
    except Exception as e:
        print(f"Error in daily suggestion: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@bp.route('/chart/weekly', methods=['GET'])
def get_weekly_chart():
    """ Get aggregated data for the last 7 days for visualization and returns averages for all metrics. """
    
    # Check DB
    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "error": "Database is disabled",
            "message": "This endpoint requires database access.",
            "status": "unavailable"
        }), 503

    # Validate User
    user_id = request.args.get('user_id')
    if not user_id or not is_valid_uuid(user_id):
        return jsonify({
            "error": "Invalid or missing user_id",
            "message": "user_id query parameter is required and must be a valid UUID.",
            "status": "bad_request"
        }), 400

    try:
        # Define Date Range (Last 7 Days including today)
        today = datetime.utcnow().date()
        end_date = datetime.combine(today, datetime.max.time())
        start_date = today - timedelta(days=6)
        
        # Query Data (Group by Date to handle multiple check-ins per day)
        # We take the AVERAGE if a user checks in multiple times a day
        query = db.session.query(
            func.date(Predictions.pred_date).label('date'),
            func.avg(Predictions.pred_score).label('mental_health_index'),
            func.avg(Predictions.sleep_hours).label('sleep_duration'),
            func.avg(Predictions.sleep_quality).label('sleep_quality'),
            func.avg(Predictions.stress_level).label('stress_level'),
            func.avg(Predictions.screen_time).label('screen_time'),
            func.avg(Predictions.productivity).label('productivity'),
            func.avg(Predictions.social).label('social_activity'),
            func.avg(Predictions.exercise).label('exercise_duration')
        ).filter(
            Predictions.user_id == uuid.UUID(user_id),
            Predictions.pred_date >= start_date,
            Predictions.pred_date <= end_date
        ).group_by(
            func.date(Predictions.pred_date)
        ).all()

        # Transform Data to Dictionary for easy lookup
        data_map = {str(row.date): row for row in query}

        # Build 7-Day Series (Fill gaps with 0)
        chart_data = []
        
        for i in range(7):
            current_day = start_date + timedelta(days=i)
            day_str = str(current_day)
            day_label = current_day.strftime('%a')
            
            # Default values if no data exists for this day
            daily_stats = {
                "date": day_str,
                "label": day_label,
                "mental_health_index": 0,
                "sleep_duration": 0,
                "sleep_quality": 0,
                "stress_level": 0,
                "screen_time": 0,
                "productivity": 0,
                "social_activity": 0,
                "exercise_duration": 0,
                "has_data": False
            }

            # If data exists, overwrite defaults
            if day_str in data_map:
                row = data_map[day_str]
                daily_stats.update({
                    "mental_health_index": round(float(row.mental_health_index or 0), 1),
                    "sleep_duration": round(float(row.sleep_duration or 0), 1),
                    "sleep_quality": round(float(row.sleep_quality or 0), 1), 
                    "stress_level": round(float(row.stress_level or 0), 1),
                    "screen_time": round(float(row.screen_time or 0), 1),
                    "productivity": round(float(row.productivity or 0), 1),
                    "social_activity": round(float(row.social_activity or 0), 1),
                    "exercise_duration": round(float(row.exercise_duration or 0), 1), 
                    "has_data": True
                })
            
            chart_data.append(daily_stats)

        return jsonify({
            "status": "success",
            "data": chart_data
        }), 200

    except Exception as e:
        print(f"Error generating chart: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

# ===================== #
#   HELPER FUNCTIONS    #
# ===================== #

def format_db_output(data):
    """Helper to transform DB flat data to nested JSON structure."""
    wellness_analysis = {
        "areas_for_improvement": [],
        "strengths": []
    }
    ai_advice_dict = {}
    
    for detail in data.get("details", []):
        entry = {"feature": detail["factor_name"], "impact_score": detail["impact_score"]}
        
        # Check factor_type safely
        f_type = detail.get("factor_type", FACTOR_TYPE_IMPROVEMENT)
        
        if f_type == FACTOR_TYPE_STRENGTH:
            wellness_analysis["strengths"].append(entry)
        else:
            wellness_analysis["areas_for_improvement"].append(entry)
            if detail.get("advices"):
                ai_advice_dict[detail["factor_name"]] = {
                    "advices": detail["advices"],
                    "references": detail["references"]
                }
    
    return {
        "prediction_score": data["prediction_score"],
        "health_level": model.categorize_mental_health_score(data["prediction_score"]),
        "wellness_analysis": wellness_analysis,
        "advice": {
            "description": data.get("ai_desc") or "Description not available.",
            "factors": ai_advice_dict
        }
    }

def process_prediction(prediction_id, json_input, created_at, app):
    """Background task for processing prediction."""
    try:
        with app.app_context():
            # Convert input to DataFrame
            if isinstance(json_input, dict):
                df = pd.DataFrame([json_input])
            else:
                df = pd.DataFrame(json_input)
            
            # Fast part: Prediction & Analysis
            prediction = model.model.predict(df)
            prediction_score = float(prediction[0])
            
            wellness_analysis = model.analyze_wellness_factors(df)
            if not wellness_analysis:
                print(f"❌ Wellness analysis failed for {prediction_id}. Using fallback.")
                wellness_analysis = {"areas_for_improvement": [], "strengths": []}
            
            mental_health_category = model.categorize_mental_health_score(prediction_score)
            
            # Store partial result
            cache.store_prediction(prediction_id, {
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
            
            # Slow part: Gemini AI
            api_key = current_app.config.get('GEMINI_API_KEY')
            ai_advice = ai.get_ai_advice(prediction_score, mental_health_category, wellness_analysis, api_key)
            
            if not ai_advice or not isinstance(ai_advice, dict):
                print(f"⚠️ AI advice generation failed for {prediction_id}. Using fallback.")
                ai_advice = {"factors": {}, "description": "AI advice could not be generated at this time."}
            
            if not current_app.config.get('DB_DISABLED', False):
                try:
                    save_to_db(prediction_id, json_input, prediction_score, wellness_analysis, ai_advice)
                except Exception as db_error:
                    print(f"⚠️ Failed to save to database: {db_error}")

            # Update with full result
            cache.update_prediction(prediction_id, {
                "status": "ready",
                "result": {
                    "prediction_score": prediction_score,
                    "health_level": mental_health_category,
                    "wellness_analysis": wellness_analysis,
                    "advice": ai_advice
                },
                "completed_at": datetime.now().isoformat()
            })
    
    except Exception as e:
        cache.update_prediction(prediction_id, {"status": "error", "error": str(e)})

def save_to_db(prediction_id, json_input, prediction_score, wellness_analysis, ai_advice):
    """
    Save prediction results AND update Daily/Weekly streaks.
    Returns: True if streak updated successfully (or no user_id), False if streak failed.
    """
    if current_app.config.get('DB_DISABLED', False):
        print("ℹ️ DB disabled, skipping save_to_db.")
        return
    
    with current_app.app_context():
        print(f"🔄 [DB] Saving data for ID: {prediction_id}...")
        
        ai_desc_text = None
        if isinstance(ai_advice, dict):
            ai_desc_text = ai_advice.get('description')

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
            pred_score=prediction_score,
            ai_desc=ai_desc_text
        )
        db.session.add(new_pred)
        db.session.flush()
        
        # Save details
        def save_detail_list(items, category_label):
            if not items: return
            for item in items:
                fname = item['feature']
                detail = PredDetails(
                    pred_id=new_pred.pred_id,
                    factor_name=fname,
                    factor_type=category_label,
                    impact_score=float(item['impact_score'])
                )
                db.session.add(detail)
                db.session.flush()
                
                # The AI advice generation (get_ai_advice) strictly targets 'areas_for_improvement'.
                # Strengths are positive attributes, so no advice or references are generated or stored for them.
                if category_label == FACTOR_TYPE_IMPROVEMENT:
                    factor_data = {}
                    if isinstance(ai_advice, dict):
                        factors_map = ai_advice.get('factors', {})
                        if fname in factors_map: factor_data = factors_map[fname]

                    for tip in factor_data.get('advices', []):
                        if tip: db.session.add(Advices(detail_id=detail.detail_id, advice_text=str(tip)))
                    for ref in factor_data.get('references', []):
                        if ref: db.session.add(References(detail_id=detail.detail_id, reference_link=str(ref)))

        save_detail_list(wellness_analysis.get('areas_for_improvement', []), FACTOR_TYPE_IMPROVEMENT)
        save_detail_list(wellness_analysis.get('strengths', []), FACTOR_TYPE_STRENGTH)

        # Update user streaks if user_id provided
        if u_id:
            try:
                client_date_str = json_input.get('local_date')

                if client_date_str:
                    try:
                        current_date = datetime.strptime(client_date_str, '%Y-%m-%d').date()
                    except ValueError:
                        print("⚠️ Invalid local_date format. Fallback to UTC.")
                        current_date = datetime.utcnow().date()
                else:
                    current_date = datetime.utcnow().date()

                with db.session.begin_nested():
                    streak_record = db.session.query(UserStreaks).filter(
                        UserStreaks.user_id == u_id
                    ).with_for_update().one_or_none()
                    
                    if not streak_record:
                        # New User: Start both streaks
                        new_streak = UserStreaks(
                            user_id=u_id,
                            curr_daily_streak=1,
                            last_daily_date=current_date,
                            curr_weekly_streak=1,
                            last_weekly_date=current_date
                        )
                        db.session.add(new_streak)
                    
                    else:
                        # --- DAILY LOGIC ---
                        last_daily = streak_record.last_daily_date

                        if last_daily is None:
                            # Fallback if null - start or re-initialize daily streak
                            streak_record.curr_daily_streak = 1
                            streak_record.last_daily_date = current_date
                        else:
                            if last_daily == current_date:
                                # Already checked in today
                                pass
                            elif last_daily == current_date - timedelta(days=1):
                                streak_record.curr_daily_streak += 1
                                streak_record.last_daily_date = current_date
                            else:
                                # Missed one or more days -> reset daily streak
                                streak_record.curr_daily_streak = 1
                                streak_record.last_daily_date = current_date

                        # --- WEEKLY LOGIC ---
                        last_weekly = streak_record.last_weekly_date
                        
                        if last_weekly:
                            # Calculate the start of the week (Monday) for both dates
                            # This handles month/year boundaries correctly
                            start_of_current_week = current_date - timedelta(days=current_date.weekday())
                            start_of_last_checkin = last_weekly - timedelta(days=last_weekly.weekday())
                            
                            days_diff = (start_of_current_week - start_of_last_checkin).days
                            
                            if days_diff == 0:
                                pass
                            elif days_diff == 7:
                                streak_record.curr_weekly_streak += 1
                                streak_record.last_weekly_date = current_date
                            else:
                                streak_record.curr_weekly_streak = 1
                                streak_record.last_weekly_date = current_date
                        else:
                            streak_record.curr_weekly_streak = 1
                            streak_record.last_weekly_date = current_date
                        
            except Exception as e:
                print(f"⚠️ Streak update failed: {e}")
                print(f"   Details: Prediction still saved to database, only streak tracking failed.")
        
        db.session.commit()
        print(f"💾 Database save completed for {prediction_id}")

def read_from_db(prediction_id=None, user_id=None):
    """Read prediction data from database."""
    from flask import current_app
    if current_app.config.get('DB_DISABLED', False):
        return {"error": "Database disabled", "status": "disabled"}

    try:
        base_query = db.select(Predictions).options(
            selectinload(Predictions.details).selectinload(PredDetails.advices),
            selectinload(Predictions.details).selectinload(PredDetails.references)
        )
        
        if prediction_id:
            try:
                pred_uuid = uuid.UUID(prediction_id)
            except ValueError:
                return {
                    "error": "Invalid prediction_id format. Must be a valid UUID string.",
                    "status": "bad_request"
                }
            
            stmt = base_query.filter(Predictions.pred_id == pred_uuid)
            pred = db.session.execute(stmt).scalar_one_or_none()
            
            if not pred:
                return {"error": "Prediction not found", "status": "not_found"}
            predictions = [pred]
        
        elif user_id:
            try:
                user_uuid = uuid.UUID(user_id)
            except ValueError:
                return {
                    "error": "Invalid user_id format. Must be a valid UUID string.",
                    "status": "bad_request"
                }
            
            stmt = base_query.filter(Predictions.user_id == user_uuid).order_by(Predictions.pred_date.desc())
            predictions = db.session.execute(stmt).scalars().all()
            
            if not predictions:
                return {"error": "No predictions found for this user", "status": "not_found"}
        else:
            return {"error": "Either prediction_id or user_id must be provided", "status": "bad_request"}
        
        # Format data
        result = []
        for pred in predictions:
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
                "ai_desc": pred.ai_desc,
                "details": []
            }
            
            for detail in pred.details:
                detail_data = {
                    "factor_name": detail.factor_name,
                    "impact_score": detail.impact_score,
                    "factor_type": detail.factor_type if detail.factor_type is not None else 'improvement',
                    "advices": [a.advice_text for a in detail.advices],
                    "references": [r.reference_link for r in detail.references]
                }
                pred_data["details"].append(detail_data)
            result.append(pred_data)
        
        if prediction_id:
            return {"status": "success", "data": result[0] if result else None}
        else:
            return {"status": "success", "data": result, "total_predictions": len(result)}
    
    except Exception as e:
        print(f"Error reading from database: {e}")
        return {"error": str(e), "status": "error"}
