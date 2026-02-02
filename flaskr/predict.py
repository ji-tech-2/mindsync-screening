"""
Prediction routes and business logic
"""
import uuid
import threading
import pandas as pd
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy.orm import selectinload

from . import model, cache, ai
from .db import db, Predictions, PredDetails, Advices, References, UserStreaks, is_valid_uuid

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
    from flask import current_app
    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "status": "not_found",
            "message": "Prediction ID not found (DB disabled)."
        }), 404

    db_result = read_from_db(prediction_id=prediction_id)

    if db_result.get("status") == "success":
        data = db_result["data"]
        
        wellness_analysis = {
            "areas_for_improvement": [],
            "strengths": []
        }
        ai_advice_dict = {}
        
        for detail in data.get("details", []):
            wellness_analysis["areas_for_improvement"].append({
                "feature": detail["factor_name"],
                "impact_score": detail["impact_score"]
            })
            ai_advice_dict[detail["factor_name"]] = {
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
                "health_level": model.categorize_mental_health_score(data["prediction_score"]),
                "wellness_analysis": wellness_analysis,
                "advice": {
                    "description": "Historical result retrieved from database.",
                    "factors": ai_advice_dict
                }
            }
        }), 200
    
    return jsonify({
        "status": "not_found",
        "message": "Prediction ID not found"
    }), 404

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
    """
    Get full history of predictions for a user (Including Advice).
    Returns list of results formatted exactly like /result endpoint.
    """
    
    # 1. Check DB Status
    if current_app.config.get('DB_DISABLED', False):
        return jsonify({
            "status": "error", 
            "message": "Database is disabled. History unavailable."
        }), 503

    # 2. Validate UUID
    if not is_valid_uuid(user_id):
        return jsonify({"error": "Invalid User ID format"}), 400
        
    try:
        # 3. Retrieve raw data (includes advice) from helper function
        db_result = read_from_db(user_id=user_id)
        
        if db_result.get("status") == "success":
            raw_list = db_result["data"]
            formatted_history = []
            
            # 4. Re-format each item to match /result schema (Nested JSON)
            for item in raw_list:
                wellness_analysis = {
                    "areas_for_improvement": [],
                    "strengths": []
                }
                ai_advice_dict = {}
                
                for detail in item.get("details", []):
                    wellness_analysis["areas_for_improvement"].append({
                        "feature": detail["factor_name"],
                        "impact_score": detail["impact_score"]
                    })
                    ai_advice_dict[detail["factor_name"]] = {
                        "advices": detail["advices"],
                        "references": detail["references"]
                    }
                
                formatted_item = {
                    "prediction_id": item["prediction_id"],
                    "prediction_score": item["prediction_score"],
                    "health_level": model.categorize_mental_health_score(item["prediction_score"]),
                    "created_at": item["prediction_date"],
                    # FULL DETAILS INCLUDED HERE:
                    "wellness_analysis": wellness_analysis,
                    "advice": {
                        "description": "Historical result retrieved from database.",
                        "factors": ai_advice_dict
                    }
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

# ===================== #
#   HELPER FUNCTIONS    #
# ===================== #

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
            from flask import current_app
            api_key = current_app.config.get('GEMINI_API_KEY')
            ai_advice = ai.get_ai_advice(prediction_score, mental_health_category, wellness_analysis, api_key)
            
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
            print(f"✅ Full result ready for {prediction_id}")
            
            # Save to database if enabled
            from flask import current_app
            if not current_app.config.get('DB_DISABLED', False):
                try:
                    save_to_db(prediction_id, json_input, prediction_score, wellness_analysis, ai_advice)
                except Exception as db_error:
                    print(f"⚠️ Failed to save to database: {db_error}")
                    cache.update_prediction(prediction_id, {
                        "status": "ready",
                        "db_save_status": "error",
                        "db_error": str(db_error)
                    })
    
    except Exception as e:
        print(f"❌ Error processing {prediction_id}: {e}")
        cache.update_prediction(prediction_id, {
            "status": "error",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })

def save_to_db(prediction_id, json_input, prediction_score, wellness_analysis, ai_advice):
    """
    Save prediction results AND update Daily/Weekly streaks.
    Returns: True if streak updated successfully (or no user_id), False if streak failed.
    """
    from flask import current_app
    if current_app.config.get('DB_DISABLED', False):
        print("ℹ️ DB disabled, skipping save_to_db.")
        return
    
    with current_app.app_context():
        print(f"🔄 [DB] Saving data for ID: {prediction_id}...")
        
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
        db.session.flush()
        
        # Save details
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
                
                # Get AI advice for this factor
                if isinstance(ai_advice, dict):
                    factor_data = ai_advice.get('factors', {}).get(fname, {})
                else:
                    factor_data = {}
                
                # Save advices
                for tip in factor_data.get('advices', []):
                    if tip:
                        db.session.add(Advices(
                            detail_id=detail.detail_id,
                            advice_text=str(tip)
                        ))
                
                # Save references
                for ref in factor_data.get('references', []):
                    if ref:
                        db.session.add(References(
                            detail_id=detail.detail_id,
                            reference_link=str(ref)
                        ))

        # Update user streaks if user_id provided
        streak_success = True

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
                print(f"⚠️ Warning: Streak update failed, but saving prediction. Error: {e}")
                streak_success = False
        
        db.session.commit()
        print(f"💾 Database save completed for {prediction_id}")
        return streak_success

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
                "details": []
            }
            
            for detail in pred.details:
                detail_data = {
                    "factor_name": detail.factor_name,
                    "impact_score": detail.impact_score,
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
