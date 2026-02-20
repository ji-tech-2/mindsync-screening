"""
Prediction routes and business logic
"""

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta

import jwt
import pandas as pd
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import func
from sqlalchemy.orm import selectinload

from . import model, cache, ai
from .db import (
    db,
    Predictions,
    PredDetails,
    Advices,
    References,
    UserStreaks,
    WeeklyCriticalFactors,
    WeeklyChartData,
    DailySuggestions,
    is_valid_uuid,
)

logger = logging.getLogger(__name__)

# Constants
FACTOR_TYPE_IMPROVEMENT = "improvement"
FACTOR_TYPE_STRENGTH = "strengths"

# ===================== #
#   AUTH HELPER FUNCS   #
# ===================== #


def _get_public_key():
    """Retrieve public key from config."""
    # Ensure newline format is correct if passed as single line string
    # Replace literal \n with actual newlines if necessary
    key = current_app.config.get("JWT_PUBLIC_KEY")
    if not key:
        logger.error("JWT_PUBLIC_KEY not configured")
        return None
    return key.replace("\\n", "\n")


def get_jwt_identity():
    """
    Extract user_id (sub) from JWT in httpOnly cookie.
    Returns user_id (str) or None if invalid/missing.
    """
    token = request.cookies.get("jwt")
    if not token:
        return None

    public_key = _get_public_key()
    if not public_key:
        return None

    try:
        # RS256 is standard for asymmetric JWT
        payload = jwt.decode(token, public_key, algorithms=["RS256"])
        return payload.get("sub")  # convention: 'sub' holds the user ID
    except jwt.ExpiredSignatureError:
        logger.debug("JWT token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token: %s", e)
        return None


def hash_ip(ip_address):
    """Create a hash of the IP address for guest identification."""
    if not ip_address:
        return None
    return hashlib.sha256(ip_address.encode("utf-8")).hexdigest()


def _check_ownership(stored_user_id, stored_guest_id, current_user_id, current_guest):
    """Check if current user owns the prediction."""
    if stored_user_id and str(stored_user_id) == str(current_user_id):
        return True
    if stored_guest_id and str(stored_guest_id) == str(current_guest):
        return True
    return False


def _build_status_response(prediction_data, status, prediction_id):
    """Build response based on prediction status."""
    if status == "processing":
        logger.debug("Prediction %s still processing", prediction_id)
        return (
            jsonify(
                {
                    "status": "processing",
                    "message": (
                        "Prediction is still being processed. "
                        "Please try again in a moment."
                    ),
                }
            ),
            202,
        )
    if status == "partial":
        logger.info("Returning partial result for %s", prediction_id)
        return (
            jsonify(
                {
                    "status": "partial",
                    "result": prediction_data["result"],
                    "message": "Prediction ready. AI advice still processing.",
                    "created_at": prediction_data["created_at"],
                }
            ),
            200,
        )
    if status == "ready":
        logger.info("Returning complete result for %s", prediction_id)
        return (
            jsonify(
                {
                    "status": "ready",
                    "result": prediction_data["result"],
                    "created_at": prediction_data["created_at"],
                    "completed_at": prediction_data["completed_at"],
                }
            ),
            200,
        )
    if status == "error":
        logger.error(
            "Prediction %s encountered an error: %s",
            prediction_id,
            str(prediction_data.get("error")),
        )
        return (
            jsonify(
                {
                    "status": "error",
                    "error": prediction_data["error"],
                    "created_at": prediction_data["created_at"],
                    "completed_at": prediction_data.get("completed_at"),
                }
            ),
            500,
        )
    return None


bp = Blueprint("predict", __name__, url_prefix="/")

# ===================== #
#   PREDICTION ROUTES   #
# ===================== #


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint - Synchronous Prediction + Asynchronous Storage.
    
    Returns prediction result immediately (200 OK) after fast model inference.
    Storage to cache and database happens in background thread.
    """
    logger.info("Received POST request to /predict")

    if model.model is None:
        logger.error("Model not loaded - cannot process prediction")
        return jsonify({"error": "Model failed to load. Check server logs."}), 500

    try:
        json_input = request.get_json()
        logger.debug(
            "Received prediction input with keys: %s",
            str(list(json_input.keys())) if json_input else "None",
        )

        # Authentication / Guest Logic
        user_id = get_jwt_identity()
        guest_id = None

        if not user_id:
            # Try to get guest_id from cookie
            guest_id = request.cookies.get("guest_id")

            if guest_id:
                # Validate that it's a proper UUID
                if not is_valid_uuid(guest_id):
                    logger.warning("Invalid guest_id format in cookie: %s", guest_id)
                    guest_id = str(uuid.uuid4())
                    logger.info("Generated new guest_id: %s", guest_id)
                else:
                    logger.info("Guest user identified from cookie: %s", guest_id)
            else:
                # No guest_id cookie found, generate new one
                guest_id = str(uuid.uuid4())
                logger.info("Generated new guest_id: %s", guest_id)
        else:
            logger.info("Authenticated user identified: %s", user_id)

        # Generate unique prediction_id
        prediction_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        start_ts = time.time()
        logger.info("Created prediction request with ID: %s", prediction_id)

        # ========================================
        # SYNCHRONOUS PREDICTION (Fast - ~20ms)
        # ========================================
        
        # Convert input to DataFrame
        if isinstance(json_input, dict):
            df = pd.DataFrame([json_input])
        else:
            df = pd.DataFrame(json_input)
        logger.debug("Input converted to DataFrame with shape %s", df.shape)

        # Run model prediction synchronously
        logger.info("Running synchronous model prediction for %s", prediction_id)
        ridge_start = time.time()
        prediction = model.model.predict(df)
        prediction_score = float(prediction[0])
        prediction_score = max(0, min(100, prediction_score))  # Clamp to [0, 100]
        ridge_prediction_time = (time.time() - ridge_start) * 1000  # Convert to ms
        
        logger.info(
            "Prediction score for %s: %.2f (inference took %.2f ms)", 
            prediction_id, prediction_score, ridge_prediction_time
        )

        # Analyze wellness factors
        logger.debug("Analyzing wellness factors for %s", prediction_id)
        wellness_analysis = model.analyze_wellness_factors(df)
        if not wellness_analysis:
            logger.warning(
                "Wellness analysis failed for %s, using fallback", prediction_id
            )
            wellness_analysis = {"areas_for_improvement": [], "strengths": []}
        else:
            logger.debug("Wellness analysis complete for %s", prediction_id)

        # Categorize mental health
        mental_health_category = model.categorize_mental_health_score(
            prediction_score
        )
        logger.info(
            f"Mental health category for {prediction_id}: {mental_health_category}"
        )

        # Calculate total processing time
        total_processing_ms = (time.time() - start_ts) * 1000

        # Build result object for immediate response
        result = {
            "prediction_score": prediction_score,
            "health_level": mental_health_category,
            "wellness_analysis": wellness_analysis,
            "timing": {
                "ridge_prediction_ms": round(ridge_prediction_time, 2),
                "total_processing_ms": round(total_processing_ms, 2),
            },
        }

        logger.info(
            "⏱️  [INSTANT RESPONSE] Total processing time: %.2f ms", 
            total_processing_ms
        )

        # ========================================
        # BACKGROUND STORAGE (Async)
        # ========================================
        
        # Get Flask app context for background thread
        app_context = current_app._get_current_object().app_context()
        
        # Start background storage thread
        logger.info("Starting background storage thread for %s", prediction_id)
        thread = threading.Thread(
            target=save_to_storage_background,
            args=(
                prediction_id,
                json_input,
                prediction_score,
                mental_health_category,
                wellness_analysis,
                created_at,
                ridge_prediction_time,
                total_processing_ms,
                user_id,
                guest_id,
                app_context,
            ),
        )
        thread.daemon = True
        thread.start()
        logger.debug("Background storage thread started for %s", prediction_id)

        # ========================================
        # INSTANT RESPONSE (200 OK)
        # ========================================
        
        return (
            jsonify(
                {
                    "prediction_id": prediction_id,
                    "status": "success",
                    "result": result,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error("Error processing prediction request: %s", e, exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 400


@bp.route("/result/<prediction_id>", methods=["GET"])
def get_result(prediction_id):
    """Check prediction status and get results."""
    logger.info("Received GET request to /result/%s", prediction_id)

    # Validate UUID
    if not is_valid_uuid(prediction_id):
        logger.warning("Invalid UUID format provided: %s", prediction_id)
        return (
            jsonify(
                {
                    "error": "Invalid ID format",
                    "message": "The provided ID must be a standard UUID.",
                }
            ),
            400,
        )

    # Get current user/guest identity
    user_id = get_jwt_identity()
    guest_hash = None
    if not user_id:
        # Try to get guest_id from cookie
        guest_hash = request.cookies.get("guest_id")
        if guest_hash and not is_valid_uuid(guest_hash):
            logger.warning("Invalid guest_id format in cookie: %s", guest_hash)
            guest_hash = None

    # Check cache first
    logger.debug("Checking cache for prediction %s", prediction_id)
    prediction_data = cache.fetch_prediction(prediction_id)

    if prediction_data:
        cached_user_id = prediction_data.get("user_id")
        cached_guest_id = prediction_data.get("guest_id")

        # Verify ownership
        if not _check_ownership(cached_user_id, cached_guest_id, user_id, guest_hash):
            if cached_user_id or cached_guest_id:
                logger.warning("Unauthorized access attempt to %s", prediction_id)
                return (
                    jsonify(
                        {
                            "error": "Unauthorized",
                            "message": "You do not have permission.",
                        }
                    ),
                    403,
                )

        status = prediction_data["status"]
        logger.debug("Cache hit for %s with status: %s", prediction_id, status)

        # Build response based on status
        response = _build_status_response(prediction_data, status, prediction_id)
        if response:
            return response

    # Fallback to database if enabled
    if current_app.config.get("DB_DISABLED", False):
        logger.warning(
            "Prediction %s not found in cache and DB disabled", prediction_id
        )
        return jsonify({"status": "not_found", "message": "Not found."}), 404

    logger.debug("Cache miss for %s, checking database", prediction_id)
    db_result = read_from_db(prediction_id=prediction_id)

    if db_result.get("status") == "success":
        logger.info("Found prediction %s in database", prediction_id)
        data = db_result["data"]
        db_user_id = data.get("user_id")
        db_guest_id = data.get("guest_id")

        # Verify ownership in DB
        if not _check_ownership(db_user_id, db_guest_id, user_id, guest_hash):
            if db_user_id or db_guest_id:
                logger.warning("Unauthorized DB access to %s", prediction_id)
                return (
                    jsonify(
                        {
                            "error": "Unauthorized",
                            "message": "You do not have permission.",
                        }
                    ),
                    403,
                )

        return (
            jsonify(
                {
                    "status": "ready",
                    "source": "database",
                    "created_at": data["prediction_date"],
                    "completed_at": data["prediction_date"],
                    "result": format_db_output(data),
                }
            ),
            200,
        )

    logger.warning("Prediction %s not found in cache or database", prediction_id)
    return jsonify(db_result), 404


@bp.route("/advice", methods=["POST"])
def advice():
    """Generate AI advice (for testing/manual calls)."""
    logger.info("Received POST request to /advice")
    try:
        json_input = request.get_json()
        logger.debug(
            "Advice request input keys:"
            + (str(list(json_input.keys())) if json_input else "None")
        )

        prediction_score = json_input.get("prediction_score")
        if prediction_score is None:
            pred_list = json_input.get("prediction")
            if pred_list and isinstance(pred_list, list):
                prediction_score = pred_list[0]
            else:
                prediction_score = pred_list

        if prediction_score is not None:
            try:
                # 1. Convert to float first (handles "95", 95, or 95.5)
                score_val = float(prediction_score)
                # 2. Apply clamp
                prediction_score = max(0, min(100, score_val))
            except (ValueError, TypeError):
                # Handle cases where data is garbage (e.g., "high")
                prediction_score = 0  # Default fallback

        category = json_input.get("mental_health_category")
        analysis = json_input.get("wellness_analysis")

        if prediction_score is None or category is None or analysis is None:
            logger.warning("Missing required inputs for advice generation")
            return (
                jsonify(
                    {"error": "Missing inputs from /predict result", "status": "error"}
                ),
                400,
            )

        logger.info(
            f"Generating advice for score: {prediction_score}, category: {category}"
        )
        api_keys_pool = current_app.config.get("GEMINI_API_KEYS")
        ai_advice = ai.get_ai_advice(
            prediction_score, category, analysis, api_keys_pool
        )

        logger.info("Advice generated successfully")
        return jsonify({"ai_advice": ai_advice, "status": "success"})

    except Exception as e:
        logger.error("Error in advice endpoint: %s", e, exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 400


@bp.route("/streak", methods=["GET"])
def get_streak_route():
    """Get sparse streak data (Daily & Weekly) for the authenticated user."""
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    logger.info("Received GET request to /streak for user %s", user_id)
    return get_streak(user_id)


def get_streak(user_id):
    """Get sparse streak data for a user (internal helper)."""
    # Original logic continues below...
    logger.info("Processing streak data for user %s", user_id)

    if current_app.config.get("DB_DISABLED", False):
        logger.warning("Streak request rejected - database disabled")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Database is disabled. Streak data unavailable.",
                }
            ),
            503,
        )

    if not is_valid_uuid(user_id):
        logger.warning("Invalid user_id format in streak request: %s", user_id)
        return (
            jsonify({"error": "Invalid user_id format. Must be a valid UUID string."}),
            400,
        )

    try:
        logger.debug("Querying predictions for user %s", user_id)
        user_uuid = uuid.UUID(user_id)

        streak_record = UserStreaks.query.filter_by(user_id=user_uuid).first()

        # Get all predictions for this user
        predictions = Predictions.query.filter_by(user_id=user_uuid).all()
        prediction_dates = {pred.pred_date.date() for pred in predictions}

        # Calculate daily data (Mon-Sun of current week)
        today = datetime.now().date()
        # Get Monday of current week (0=Monday, 6=Sunday)
        days_since_monday = today.weekday()
        monday = today - timedelta(days=days_since_monday)

        daily_data = []
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i in range(7):
            current_day = monday + timedelta(days=i)
            daily_data.append(
                {
                    "date": current_day.isoformat(),
                    "label": day_names[i],
                    "has_screening": current_day in prediction_dates,
                }
            )

        # Calculate weekly data (last 7 weeks)
        # Week is defined as Monday-Sunday
        weekly_data = []
        for week_offset in range(6, -1, -1):  # 6 weeks ago to this week
            week_monday = monday - timedelta(weeks=week_offset)
            week_sunday = week_monday + timedelta(days=6)

            # Check if any prediction exists in this week
            has_screening = any(
                week_monday <= date <= week_sunday for date in prediction_dates
            )

            # Format: "Jan 6-12" or "Dec 30 - Jan 5" for cross-month weeks
            if week_monday.month == week_sunday.month:
                week_label = (
                    f"{week_monday.strftime('%b')} {week_monday.day}-{week_sunday.day}"
                )
            else:
                week_label = (
                    f"{week_monday.strftime('%b')} {week_monday.day} - "
                    f"{week_sunday.strftime('%b')} {week_sunday.day}"
                )

            weekly_data.append(
                {
                    "week_start": week_monday.isoformat(),
                    "week_end": week_sunday.isoformat(),
                    "label": week_label,
                    "has_screening": has_screening,
                }
            )

        # Calculate current daily streak with grace period:
        # Streak persists if last screening was yesterday, resets if gap > 1 day
        current_daily_streak = 0
        last_daily_date = None

        # Find the most recent consecutive streak ending today or yesterday
        if today in prediction_dates:
            # Today has screening - count backward from today
            check_date = today
            last_daily_date = today
            while check_date in prediction_dates:
                current_daily_streak += 1
                check_date -= timedelta(days=1)
        elif (today - timedelta(days=1)) in prediction_dates:
            # Yesterday has screening but not today - count from yesterday (grace period)
            check_date = today - timedelta(days=1)
            last_daily_date = today - timedelta(days=1)
            while check_date in prediction_dates:
                current_daily_streak += 1
                check_date -= timedelta(days=1)
        # else: no recent screening = streak stays 0

        # Calculate current weekly streak with grace period:
        # Streak persists if last screening was this week or last week, resets if gap > 1 week
        current_weekly_streak = 0
        last_weekly_date = None

        def _has_screening_in_week(week_monday):
            week_sunday = week_monday + timedelta(days=6)
            return any(week_monday <= date <= week_sunday for date in prediction_dates)

        if _has_screening_in_week(monday):
            # This week has screening - count backward from this week
            check_week_monday = monday
            last_weekly_date = monday
            while _has_screening_in_week(check_week_monday):
                current_weekly_streak += 1
                check_week_monday -= timedelta(weeks=1)
        elif _has_screening_in_week(monday - timedelta(weeks=1)):
            # Last week has screening but not this week - count from last week (grace period)
            check_week_monday = monday - timedelta(weeks=1)
            last_weekly_date = monday - timedelta(weeks=1)
            while _has_screening_in_week(check_week_monday):
                current_weekly_streak += 1
                check_week_monday -= timedelta(weeks=1)
        # else: no recent screening = streak stays 0

        logger.info("Streak data processed for user %s", user_id)
        return (
            jsonify(
                {
                    "status": "success",
                    "data": {
                        "user_id": user_id,
                        "current_streak": {
                            "daily": current_daily_streak,
                            "daily_last_date": (
                                last_daily_date.isoformat() if last_daily_date else None
                            ),
                            "weekly": current_weekly_streak,
                            "weekly_last_date": (
                                last_weekly_date.isoformat()
                                if last_weekly_date
                                else None
                            ),
                        },
                        "daily": daily_data,
                        "weekly": weekly_data,
                    },
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(
            "Error retrieving streak for user %s: %s", user_id, e, exc_info=True
        )
        return jsonify({"error": str(e), "status": "error"}), 500


@bp.route("/history", methods=["GET"])
def get_history():
    """Get full history of predictions for the authenticated user."""
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    logger.info("Received GET request to /history for user %s", user_id)

    # Check DB Status
    if current_app.config.get("DB_DISABLED", False):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Database is disabled. History unavailable.",
                }
            ),
            503,
        )

    # Use existing helper but modified to accept user_id
    # Wait, read_from_db logic was updated (or SHOULD be updated)

    try:
        db_result = read_from_db(user_id=user_id)
        if db_result.get("status") == "success":
            # Format list of results
            formatted_history = []
            for item in db_result["data"]:
                formatted_data = format_db_output(item)

                # Strip user_id/guest_id info from public item history response if not needed
                # item["input_data"] is already there?
                # format_db_output structures it nicely

                formatted_item = {
                    "prediction_id": item["prediction_id"],
                    "prediction_score": item["prediction_score"],
                    "health_level": model.categorize_mental_health_score(
                        item["prediction_score"]
                    ),
                    "created_at": item["prediction_date"],
                    "wellness_analysis": formatted_data["wellness_analysis"],
                    "advice": formatted_data["advice"],
                }
                formatted_history.append(formatted_item)

            return (
                jsonify(
                    {
                        "status": "success",
                        "count": len(formatted_history),
                        "data": formatted_history,
                    }
                ),
                200,
            )

        elif db_result.get("status") == "not_found":
            # User has no history yet, return empty list (not error)
            return jsonify({"status": "success", "count": 0, "data": []}), 200

        else:
            return jsonify(db_result), 400

    except Exception as e:
        logger.error("Error fetching history: %s", e, exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500


@bp.route("/weekly-critical-factors", methods=["GET"])
def get_weekly_critical_factors():
    """
    Get the top 3 most frequent areas_for_improvement from the last week.
    Also generates AI advice for these critical factors.

    Query params:
        days (optional): Number of days to look back (default: 7)
    """
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    days = request.args.get("days", 7, type=int)

    # Check if database is enabled
    if current_app.config.get("DB_DISABLED", False):
        return (
            jsonify(
                {
                    "error": "Database is disabled",
                    "message": "This endpoint requires database access.",
                    "status": "unavailable",
                }
            ),
            503,
        )

    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        week_start = start_date.date()
        week_end = end_date.date()

        # Check if cached data exists for this time period
        cached_query = WeeklyCriticalFactors.query.filter_by(
            week_start=week_start,
            week_end=week_end,
            days=days,
            user_id=uuid.UUID(user_id),
        )

        cached_data = cached_query.first()

        if cached_data:
            # Return cached data
            return (
                jsonify({"status": "success", "cached": True, **cached_data.to_dict()}),
                200,
            )

        # No cache found, calculate fresh data
        # Build query to get top critical factors
        query = (
            db.session.query(
                PredDetails.factor_name,
                func.count(PredDetails.factor_name).label(
                    "occurrence_count"
                ),  # pylint: disable=not-callable
                func.avg(PredDetails.impact_score).label("avg_impact_score"),
            )
            .join(Predictions, Predictions.pred_id == PredDetails.pred_id)
            .filter(
                Predictions.pred_date >= start_date,
                Predictions.pred_date <= end_date,
                PredDetails.impact_score
                > 0,  # Only areas for improvement (positive impact)
            )
        )

        # Filter by user if provided
        if user_id:
            query = query.filter(Predictions.user_id == uuid.UUID(user_id))

        # Group by factor and order by frequency
        results = (
            query.group_by(PredDetails.factor_name)
            .order_by(
                func.count(PredDetails.factor_name).desc()
            )  # pylint: disable=not-callable
            .limit(3)
            .all()
        )

        # Format the results
        top_factors = []
        for row in results:
            top_factors.append(
                {
                    "factor_name": row.factor_name,
                    "count": row.occurrence_count,
                    "avg_impact_score": (
                        float(row.avg_impact_score) if row.avg_impact_score else 0.0
                    ),
                }
            )

        # Get additional stats
        total_predictions = db.session.query(
            func.count(Predictions.pred_id)
        ).filter(  # pylint: disable=not-callable
            Predictions.pred_date >= start_date, Predictions.pred_date <= end_date
        )

        if user_id:
            total_predictions = total_predictions.filter(
                Predictions.user_id == uuid.UUID(user_id)
            )

        total_count = total_predictions.scalar() or 0

        # Generate AI advice for these critical factors
        api_keys_pool = current_app.config.get("GEMINI_API_KEYS")
        ai_advice = None

        if top_factors and api_keys_pool:
            ai_advice = ai.get_weekly_advice(top_factors, api_keys_pool)
        elif not api_keys_pool:
            ai_advice = {
                "description": "AI advice unavailable (API key not configured)",
                "factors": {},
            }
        elif not top_factors:
            ai_advice = {
                "description": (
                    "No wellness data recorded in this period. Complete "
                    "wellness checks to get personalized weekly insights!"
                ),
                "factors": {},
            }

        # Only cache if there's actual prediction data
        if top_factors and api_keys_pool:
            new_cache = WeeklyCriticalFactors(
                user_id=uuid.UUID(user_id) if user_id else None,
                week_start=week_start,
                week_end=week_end,
                days=days,
                total_predictions=total_count,
                top_factors=top_factors,
                ai_advice=ai_advice,
            )
            db.session.add(new_cache)
            db.session.commit()

        return (
            jsonify(
                {
                    "status": "success",
                    "cached": False,
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days,
                    },
                    "stats": {"total_predictions": total_count, "user_id": user_id},
                    "top_critical_factors": top_factors,
                    "advice": ai_advice,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error("Error in weekly critical factors: %s", e, exc_info=True)
        try:
            db.session.rollback()
        except Exception:
            pass
        finally:
            try:
                db.session.remove()
            except Exception:
                pass
        return jsonify({"error": str(e), "status": "error"}), 500


@bp.route("/daily-suggestion", methods=["GET"])
def get_daily_suggestion():
    """
    Get AI-powered daily suggestion based on today's areas of improvement.
    """
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    # Check if database is enabled
    if current_app.config.get("DB_DISABLED", False):
        return (
            jsonify(
                {
                    "error": "Database is disabled",
                    "message": "This endpoint requires database access.",
                    "status": "unavailable",
                }
            ),
            503,
        )

    try:
        # Validate user_id
        if not is_valid_uuid(user_id):
            return (
                jsonify(
                    {
                        "error": "Invalid user_id format",
                        "message": "user_id must be a valid UUID.",
                        "status": "bad_request",
                    }
                ),
                400,
            )

        # Calculate today's date range (midnight to midnight) in UTC
        # pred_date is stored using datetime.utcnow, so we must query in UTC
        today = datetime.utcnow().date()

        # Check if cached data exists for today
        cached_data = DailySuggestions.query.filter_by(
            user_id=uuid.UUID(user_id), date=today
        ).first()

        if cached_data:
            # Return cached data
            return (
                jsonify({"status": "success", "cached": True, **cached_data.to_dict()}),
                200,
            )

        # No cache found, calculate fresh data
        start_of_day = datetime.combine(today, datetime.min.time())
        end_of_day = datetime.combine(today, datetime.max.time())

        # Query today's predictions for this user
        results = (
            db.session.query(
                PredDetails.factor_name,
                func.avg(PredDetails.impact_score).label("avg_impact_score"),
            )
            .join(Predictions, Predictions.pred_id == PredDetails.pred_id)
            .filter(
                Predictions.user_id == uuid.UUID(user_id),
                Predictions.pred_date >= start_of_day,
                Predictions.pred_date <= end_of_day,
                PredDetails.impact_score > 0,  # Only areas for improvement
            )
            .group_by(PredDetails.factor_name)
            .order_by(func.avg(PredDetails.impact_score).desc())
            .limit(3)
            .all()
        )

        # Format the results
        top_factors = []
        for row in results:
            top_factors.append(
                {
                    "factor_name": row.factor_name,
                    "impact_score": (
                        float(row.avg_impact_score) if row.avg_impact_score else 0.0
                    ),
                }
            )

        # Get today's prediction count
        prediction_count = (
            db.session.query(
                func.count(Predictions.pred_id)
            )  # pylint: disable=not-callable
            .filter(
                Predictions.user_id == uuid.UUID(user_id),
                Predictions.pred_date >= start_of_day,
                Predictions.pred_date <= end_of_day,
            )
            .scalar()
            or 0
        )

        # Generate AI advice for today's factors
        api_keys_pool = current_app.config.get("GEMINI_API_KEYS")
        ai_advice = None

        if top_factors and api_keys_pool:
            ai_advice = ai.get_daily_advice(top_factors, api_keys_pool)
        elif not api_keys_pool:
            ai_advice = {
                "message": (
                    "AI advice unavailable. Take a moment to reflect on "
                    "your wellness today."
                )
            }
        elif prediction_count > 0 and not top_factors:
            ai_advice = {
                "message": (
                    "Great job! All your wellness areas look good today. " "Keep it up!"
                )
            }
        else:
            ai_advice = {
                "message": (
                    "No check-ins yet today. Complete a wellness check to "
                    "get personalized suggestions!"
                )
            }

        # Only cache if there's actual prediction data
        if (top_factors and api_keys_pool) or (
            prediction_count > 0 and not top_factors
        ):
            new_cache = DailySuggestions(
                user_id=uuid.UUID(user_id),
                date=today,
                prediction_count=prediction_count,
                top_factors=top_factors,
                ai_advice=ai_advice,
            )
            db.session.add(new_cache)
            db.session.commit()

        return (
            jsonify(
                {
                    "status": "success",
                    "cached": False,
                    "date": today.isoformat(),
                    "user_id": user_id,
                    "stats": {"predictions_today": prediction_count},
                    "areas_of_improvement": top_factors,
                    "suggestion": ai_advice,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error("Error in daily suggestion: %s", e, exc_info=True)
        try:
            db.session.rollback()
        except Exception:
            pass
        finally:
            try:
                db.session.remove()
            except Exception:
                pass
        return jsonify({"error": str(e), "status": "error"}), 500


@bp.route("/weekly-chart-data", methods=["GET"])
def get_weekly_chart_data():
    """
    Get weekly chart data for the authenticated user (last 7 days).
    """
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    # Check DB
    if current_app.config.get("DB_DISABLED", False):
        return (
            jsonify(
                {
                    "error": "Database is disabled",
                    "message": "This endpoint requires database access.",
                    "status": "unavailable",
                }
            ),
            503,
        )

    # Validate User
    if not is_valid_uuid(user_id):
        return (
            jsonify(
                {
                    "error": "Invalid user_id format",
                    "message": "user_id must be a valid UUID.",
                    "status": "bad_request",
                }
            ),
            400,
        )

    try:
        # Define Date Range (Last 7 Days including today)
        today = datetime.utcnow().date()
        start_date = today - timedelta(days=6)
        week_start = start_date
        week_end = today

        # Check if cached data exists for this time period
        cached_data = WeeklyChartData.query.filter_by(
            user_id=uuid.UUID(user_id), week_start=week_start, week_end=week_end
        ).first()

        if cached_data:
            # Return cached data
            return (
                jsonify({"status": "success", "cached": True, **cached_data.to_dict()}),
                200,
            )

        # No cache found, calculate fresh data
        end_date = datetime.combine(today, datetime.max.time())

        # Query Data (Group by Date to handle multiple check-ins per day)
        # We take the AVERAGE if a user checks in multiple times a day
        query = (
            db.session.query(
                func.date(Predictions.pred_date).label("date"),
                func.avg(Predictions.pred_score).label("mental_health_index"),
                func.avg(Predictions.sleep_hours).label("sleep_duration"),
                func.avg(Predictions.sleep_quality).label("sleep_quality"),
                func.avg(Predictions.stress_level).label("stress_level"),
                func.avg(Predictions.work_screen).label("work_screen"),
                func.avg(Predictions.leisure_screen).label("leisure_screen"),
                func.avg(Predictions.productivity).label("productivity"),
                func.avg(Predictions.social).label("social_activity"),
                func.avg(Predictions.exercise).label("exercise_duration"),
            )
            .filter(
                Predictions.user_id == uuid.UUID(user_id),
                Predictions.pred_date >= start_date,
                Predictions.pred_date <= end_date,
            )
            .group_by(func.date(Predictions.pred_date))
            .all()
        )

        # Transform Data to Dictionary for easy lookup
        data_map = {str(row.date): row for row in query}

        # Build 7-Day Series (Fill gaps with 0)
        chart_data = []

        for i in range(7):
            current_day = start_date + timedelta(days=i)
            day_str = str(current_day)
            day_label = current_day.strftime("%a")

            # Default values if no data exists for this day
            daily_stats = {
                "date": day_str,
                "label": day_label,
                "mental_health_index": 0,
                "sleep_duration": 0,
                "sleep_quality": 0,
                "stress_level": 0,
                "work_screen": 0,
                "leisure_screen": 0,
                "productivity": 0,
                "social_activity": 0,
                "exercise_duration": 0,
                "has_data": False,
            }

            # If data exists, overwrite defaults
            if day_str in data_map:
                row = data_map[day_str]
                daily_stats.update(
                    {
                        "mental_health_index": round(
                            float(row.mental_health_index or 0), 1
                        ),
                        "sleep_duration": round(float(row.sleep_duration or 0), 1),
                        "sleep_quality": round(float(row.sleep_quality or 0), 1),
                        "stress_level": round(float(row.stress_level or 0), 1),
                        "work_screen": round(float(row.work_screen or 0), 1),
                        "leisure_screen": round(float(row.leisure_screen or 0), 1),
                        "productivity": round(float(row.productivity or 0), 1),
                        "social_activity": round(float(row.social_activity or 0), 1),
                        "exercise_duration": round(
                            float(row.exercise_duration or 0), 1
                        ),
                        "has_data": True,
                    }
                )

            chart_data.append(daily_stats)

        # Only cache if there's actual prediction data
        has_data = any(day["has_data"] for day in chart_data)
        if has_data:
            new_cache = WeeklyChartData(
                user_id=uuid.UUID(user_id),
                week_start=week_start,
                week_end=week_end,
                chart_data=chart_data,
            )
            db.session.add(new_cache)
            db.session.commit()

        return jsonify({"status": "success", "cached": False, "data": chart_data}), 200

    except Exception as e:
        logger.error("Error in weekly chart: %s", e, exc_info=True)
        try:
            db.session.rollback()
        except Exception:
            pass
        finally:
            try:
                db.session.remove()
            except Exception:
                pass
        return jsonify({"error": str(e), "status": "error"}), 500


# ===================== #
#   HELPER FUNCTIONS    #
# ===================== #


def format_db_output(data):
    """Helper to transform DB flat data to nested JSON structure."""
    wellness_analysis = {"areas_for_improvement": [], "strengths": []}
    ai_advice_dict = {}

    for detail in data.get("details", []):
        entry = {
            "feature": detail["factor_name"],
            "impact_score": detail["impact_score"],
        }

        # Check factor_type safely
        f_type = detail.get("factor_type", FACTOR_TYPE_IMPROVEMENT)

        if f_type == FACTOR_TYPE_STRENGTH:
            wellness_analysis["strengths"].append(entry)
        else:
            wellness_analysis["areas_for_improvement"].append(entry)
            if detail.get("advices"):
                ai_advice_dict[detail["factor_name"]] = {
                    "advices": detail["advices"],
                    "references": detail["references"],
                }

    return {
        "prediction_score": data["prediction_score"],
        "health_level": model.categorize_mental_health_score(data["prediction_score"]),
        "wellness_analysis": wellness_analysis,
        "advice": {
            "description": data.get("ai_desc") or "Description not available.",
            "factors": ai_advice_dict,
        },
    }


def save_to_storage_background(
    prediction_id,
    json_input,
    prediction_score,
    mental_health_category,
    wellness_analysis,
    created_at,
    ridge_prediction_time,
    total_processing_ms,
    user_id,
    guest_id,
    app_context,
):
    """
    Background worker to save prediction results to cache and database.
    
    This function runs in a separate thread and handles:
    1. Storing prediction result to Valkey (cache)
    2. Saving prediction to Postgres (database)
    3. Generating AI advice (if user is authenticated)
    4. Updating cache with AI advice when complete
    
    Args:
        prediction_id: Unique prediction identifier
        json_input: Original input data
        prediction_score: ML model prediction score
        mental_health_category: Categorized health level
        wellness_analysis: Wellness factors analysis
        created_at: Timestamp of prediction creation
        ridge_prediction_time: Time taken for model inference (ms)
        total_processing_ms: Total synchronous processing time (ms)
        user_id: Authenticated user ID (if any)
        guest_id: Guest user ID (if any)
        app_context: Flask app context for database/cache operations
    """
    try:
        with app_context:
            logger.info(
                "⚙️  [BACKGROUND] Starting storage operations for %s", prediction_id
            )
            storage_start = time.time()

            # ========================================
            # 1. SAVE TO CACHE (Valkey)
            # ========================================
            
            try:
                logger.debug("Storing prediction result to cache for %s", prediction_id)
                cache.store_prediction(
                    prediction_id,
                    {
                        "status": "success",
                        "result": {
                            "prediction_score": prediction_score,
                            "health_level": mental_health_category,
                            "wellness_analysis": wellness_analysis,
                            "advice": None,  # Will be updated after AI processing
                            "timing": {
                                "ridge_prediction_ms": round(ridge_prediction_time, 2),
                                "total_processing_ms": round(total_processing_ms, 2),
                            },
                        },
                        "created_at": created_at,
                        "user_id": user_id,
                        "guest_id": guest_id,
                    },
                )
                logger.info("✅ Cache storage completed for %s", prediction_id)
            except Exception as cache_error:
                logger.error(
                    "❌ Failed to store prediction %s to cache: %s",
                    prediction_id,
                    cache_error,
                    exc_info=True,
                )

            # ========================================
            # 2. GENERATE AI ADVICE (if authenticated)
            # ========================================
            
            ai_advice = None
            if guest_id and not user_id:
                logger.info(
                    "Guest user detected for %s - Skipping AI advice generation",
                    prediction_id,
                )
                ai_advice = {
                    "factors": {},
                    "description": "Log in to view personalized AI advice.",
                }
            else:
                logger.info("Generating AI advice for %s", prediction_id)
                try:
                    api_keys_pool = current_app.config.get("GEMINI_API_KEYS")
                    ai_advice = ai.get_ai_advice(
                        prediction_score,
                        mental_health_category,
                        wellness_analysis,
                        api_keys_pool,
                    )

                    if not ai_advice or not isinstance(ai_advice, dict):
                        logger.warning(
                            "AI advice generation failed for %s, using fallback",
                            prediction_id,
                        )
                        ai_advice = {
                            "factors": {},
                            "description": "AI advice could not be generated at this time.",
                        }
                    else:
                        logger.info(
                            "✅ AI advice generated successfully for %s", prediction_id
                        )
                except Exception as ai_error:
                    logger.error(
                        "❌ AI advice generation error for %s: %s",
                        prediction_id,
                        ai_error,
                        exc_info=True,
                    )
                    ai_advice = {
                        "factors": {},
                        "description": "AI advice could not be generated at this time.",
                    }

            # ========================================
            # 3. SAVE TO DATABASE (Postgres)
            # ========================================
            
            if not current_app.config.get("DB_DISABLED", False):
                try:
                    logger.debug("Saving prediction %s to database", prediction_id)
                    save_to_db(
                        prediction_id,
                        json_input,
                        prediction_score,
                        wellness_analysis,
                        ai_advice,
                        user_id,
                        guest_id,
                    )
                    logger.info(
                        "✅ Database save completed for %s", prediction_id
                    )
                except Exception as db_error:
                    logger.error(
                        "❌ Failed to save prediction %s to database: %s",
                        prediction_id,
                        db_error,
                        exc_info=True,
                    )
            else:
                logger.debug("Database disabled, skipping DB save for %s", prediction_id)

            # ========================================
            # 4. UPDATE CACHE WITH AI ADVICE
            # ========================================
            
            try:
                logger.debug("Updating cache with AI advice for %s", prediction_id)
                cache.update_prediction(
                    prediction_id,
                    {
                        "status": "ready",
                        "result": {
                            "prediction_score": prediction_score,
                            "health_level": mental_health_category,
                            "wellness_analysis": wellness_analysis,
                            "advice": ai_advice,
                            "timing": {
                                "ridge_prediction_ms": round(ridge_prediction_time, 2),
                                "total_processing_ms": round(total_processing_ms, 2),
                            },
                        },
                        "completed_at": datetime.now().isoformat(),
                    },
                )
                logger.info("✅ Cache updated with AI advice for %s", prediction_id)
            except Exception as cache_update_error:
                logger.error(
                    "❌ Failed to update cache with AI advice for %s: %s",
                    prediction_id,
                    cache_update_error,
                    exc_info=True,
                )

            storage_time = (time.time() - storage_start) * 1000
            logger.info(
                "⏱️  [BACKGROUND] Total storage time for %s: %.2f ms",
                prediction_id,
                storage_time,
            )

    except Exception as e:
        logger.error(
            "❌ [BACKGROUND] Critical error in storage worker for %s: %s",
            prediction_id,
            e,
            exc_info=True,
        )


def process_prediction(
    prediction_id, json_input, created_at, start_ts, app, user_id, guest_id
):
    """Background task for processing prediction."""
    logger.info("Background processing started for prediction %s", prediction_id)
    try:
        with app.app_context():
            # Convert input to DataFrame
            if isinstance(json_input, dict):
                df = pd.DataFrame([json_input])
            else:
                df = pd.DataFrame(json_input)
            logger.debug("Input converted to DataFrame with shape %s", df.shape)

            # BENCHMARK: Start timing from model.pkl load/prediction to frontend
            total_start = time.time()

            # Fast part: Prediction & Analysis
            logger.info("Running model prediction for %s", prediction_id)
            ridge_start = time.time()
            prediction = model.model.predict(df)
            prediction_score = float(prediction[0])
            prediction_score = max(0, min(100, prediction_score))  # Clamp to [0, 100]
            ridge_prediction_time = (time.time() - ridge_start) * 1000  # Convert to ms

            logger.info(
                "Prediction score for %s: %.2f", prediction_id, prediction_score
            )
            logger.info(
                "⏱️  [BENCHMARK] Ridge prediction only: %.2f ms", ridge_prediction_time
            )

            logger.debug("Analyzing wellness factors for %s", prediction_id)
            wellness_analysis = model.analyze_wellness_factors(df)
            if not wellness_analysis:
                logger.warning(
                    "Wellness analysis failed for %s, using fallback", prediction_id
                )
                wellness_analysis = {"areas_for_improvement": [], "strengths": []}
            else:
                logger.debug("Wellness analysis complete for %s", prediction_id)

            mental_health_category = model.categorize_mental_health_score(
                prediction_score
            )
            logger.info(
                f"Mental health category for {prediction_id}: {mental_health_category}"
            )

            # Calculate server processing time from request start to data ready
            server_processing_ms = (time.time() - start_ts) * 1000

            # Store partial result
            logger.debug("Storing partial result for %s", prediction_id)
            cache.store_prediction(
                prediction_id,
                {
                    "status": "partial",
                    "result": {
                        "prediction_score": prediction_score,
                        "health_level": mental_health_category,
                        "wellness_analysis": wellness_analysis,
                        "advice": None,
                        "timing": {
                            "ridge_prediction_ms": round(ridge_prediction_time, 2),
                            "server_processing_ms": round(server_processing_ms, 2),
                            "start_timestamp": start_ts,
                        },
                    },
                    "created_at": (
                        created_at if created_at else datetime.now().isoformat()
                    ),
                    "user_id": user_id,
                    "guest_id": guest_id,
                },
            )

            # BENCHMARK: End timing - sent to cache (frontend access)
            total_time_to_frontend = (time.time() - total_start) * 1000
            logger.info("Partial result stored and ready for %s", prediction_id)
            logger.info(
                "⏱️  [BENCHMARK] Model.pkl → Data available to frontend: %.2f ms",
                total_time_to_frontend,
            )

            # Update the timing in cache
            cache.update_prediction(
                prediction_id,
                {
                    "result": {
                        "prediction_score": prediction_score,
                        "health_level": mental_health_category,
                        "wellness_analysis": wellness_analysis,
                        "advice": None,
                        "timing": {
                            "ridge_prediction_ms": round(ridge_prediction_time, 2),
                            "server_processing_ms": round(server_processing_ms, 2),
                            "start_timestamp": start_ts,
                        },
                    },
                },
            )

            # Slow part: Gemini AI - ONLY IF NOT GUEST
            ai_advice = None
            if guest_id and not user_id:
                logger.info(
                    "Guest user detected for %s - Skipping Gemini API call",
                    prediction_id,
                )
                ai_advice = {
                    "factors": {},
                    "description": "Log in to view personalized AI advice.",
                }
            else:
                logger.info("Requesting AI advice for %s", prediction_id)
                api_keys_pool = current_app.config.get("GEMINI_API_KEYS")
                ai_advice = ai.get_ai_advice(
                    prediction_score,
                    mental_health_category,
                    wellness_analysis,
                    api_keys_pool,
                )

                if not ai_advice or not isinstance(ai_advice, dict):
                    logger.warning(
                        "AI advice generation failed for %s, using fallback",
                        prediction_id,
                    )
                    ai_advice = {
                        "factors": {},
                        "description": "AI advice could not be generated at this time.",
                    }
                else:
                    logger.info(
                        "AI advice generated successfully for %s", prediction_id
                    )

            if not current_app.config.get("DB_DISABLED", False):
                try:
                    logger.debug("Saving prediction %s to database", prediction_id)
                    save_to_db(
                        prediction_id,
                        json_input,
                        prediction_score,
                        wellness_analysis,
                        ai_advice,
                        user_id,
                        guest_id,
                    )
                    logger.info(
                        f"Prediction {prediction_id} saved to database successfully"
                    )
                except Exception as db_error:
                    logger.error(
                        "Failed to save prediction %s to database: %s",
                        prediction_id,
                        db_error,
                        exc_info=True,
                    )

            # Update with full result
            logger.debug("Updating cache with final result for %s", prediction_id)
            cache.update_prediction(
                prediction_id,
                {
                    "status": "ready",
                    "result": {
                        "prediction_score": prediction_score,
                        "health_level": mental_health_category,
                        "wellness_analysis": wellness_analysis,
                        "advice": ai_advice,
                        "timing": {
                            "ridge_prediction_ms": round(ridge_prediction_time, 2),
                            "server_processing_ms": round(server_processing_ms, 2),
                            "start_timestamp": start_ts,
                        },
                    },
                    "completed_at": datetime.now().isoformat(),
                },
            )
            logger.info("Prediction %s fully processed and ready", prediction_id)

    except Exception as e:
        logger.error(
            "Error processing prediction %s: %s", prediction_id, e, exc_info=True
        )
        cache.update_prediction(prediction_id, {"status": "error", "error": str(e)})


def _save_detail_records(pred_id, items, category_label, ai_advice):
    """Helper function to save prediction detail records."""
    if not items:
        return
    logger.debug("Saving %d %s detail records", len(items), category_label)
    for item in items:
        fname = item["feature"]
        detail = PredDetails(
            pred_id=pred_id,
            factor_name=fname,
            factor_type=category_label,
            impact_score=float(item["impact_score"]),
        )
        db.session.add(detail)
        db.session.flush()

        # The AI advice generation (get_ai_advice) strictly targets
        # 'areas_for_improvement'. Strengths are positive attributes,
        # so no advice or references are generated or stored for them.
        if category_label == FACTOR_TYPE_IMPROVEMENT:
            factor_data = {}
            if isinstance(ai_advice, dict):
                factors_map = ai_advice.get("factors", {})
                if fname in factors_map:
                    factor_data = factors_map[fname]

            for tip in factor_data.get("advices", []):
                if tip:
                    db.session.add(
                        Advices(detail_id=detail.detail_id, advice_text=str(tip))
                    )
            for ref in factor_data.get("references", []):
                if ref:
                    db.session.add(
                        References(detail_id=detail.detail_id, reference_link=str(ref))
                    )


def _update_daily_streak(streak_record, current_date):
    """Helper function to update daily streak logic."""
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


def _update_weekly_streak(streak_record, current_date):
    """Helper function to update weekly streak logic."""
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


def _parse_current_date(client_date_str):
    """Parse date from client or fallback to UTC."""
    if client_date_str:
        try:
            return datetime.strptime(client_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("⚠️ Invalid local_date format. Fallback to UTC.")
            return datetime.utcnow().date()
    else:
        return datetime.utcnow().date()


def _update_user_streaks(user_id, current_date):
    """Helper function to update user streak records."""
    with db.session.begin_nested():
        streak_record = (
            db.session.query(UserStreaks)
            .filter(UserStreaks.user_id == user_id)
            .with_for_update()
            .one_or_none()
        )

        if not streak_record:
            # New User: Start both streaks
            new_streak = UserStreaks(
                user_id=user_id,
                curr_daily_streak=1,
                last_daily_date=current_date,
                curr_weekly_streak=1,
                last_weekly_date=current_date,
            )
            db.session.add(new_streak)
        else:
            _update_daily_streak(streak_record, current_date)
            _update_weekly_streak(streak_record, current_date)

    db.session.flush()


def save_to_db(
    prediction_id,
    json_input,
    prediction_score,
    wellness_analysis,
    ai_advice,
    user_id=None,
    guest_id=None,
):
    """
    Save prediction results AND update Daily/Weekly streaks.
    Returns: True if streak updated successfully (or no user_id), False if
    streak failed.
    """
    if current_app.config.get("DB_DISABLED", False):
        logger.debug("DB disabled, skipping save_to_db")
        return

    with current_app.app_context():
        logger.info("Saving prediction %s to database", prediction_id)

        ai_desc_text = None
        if isinstance(ai_advice, dict):
            ai_desc_text = ai_advice.get("description")

        # Use passed user_id or guest_id
        u_id = None
        if user_id:
            try:
                u_id = uuid.UUID(str(user_id))
            except ValueError:
                logger.warning("Invalid user_id provided: %s", user_id)

        g_id = None
        if guest_id:
            try:
                g_id = uuid.UUID(str(guest_id))
            except ValueError:
                logger.warning("Invalid guest_id provided: %s", guest_id)

        logger.debug("Creating prediction record. User: %s, Guest: %s", u_id, g_id)

        work_screen_val = float(json_input.get("work_screen_hours", 0))
        leisure_screen_val = float(json_input.get("leisure_screen_hours", 0))
        screen_time_val = float(
            json_input.get("screen_time_hours")
            if json_input.get("screen_time_hours") is not None
            else work_screen_val + leisure_screen_val
        )

        new_pred = Predictions(
            pred_id=uuid.UUID(prediction_id),
            user_id=u_id,
            guest_id=g_id,
            screen_time=screen_time_val,
            work_screen=work_screen_val,
            leisure_screen=leisure_screen_val,
            sleep_hours=float(json_input.get("sleep_hours", 0)),
            stress_level=float(json_input.get("stress_level_0_10", 0)),
            productivity=float(json_input.get("productivity_0_100", 0)),
            social=float(json_input.get("social_hours_per_week", 0)),
            sleep_quality=int(json_input.get("sleep_quality_1_5", 0)),
            exercise=int(json_input.get("exercise_minutes_per_week", 0)),
            pred_score=prediction_score,
            ai_desc=ai_desc_text,
        )
        db.session.add(new_pred)
        db.session.flush()
        logger.debug("Prediction record created with ID: %s", prediction_id)

        # Save details
        _save_detail_records(
            new_pred.pred_id,
            wellness_analysis.get("areas_for_improvement", []),
            FACTOR_TYPE_IMPROVEMENT,
            ai_advice,
        )
        _save_detail_records(
            new_pred.pred_id,
            wellness_analysis.get("strengths", []),
            FACTOR_TYPE_STRENGTH,
            ai_advice,
        )
        logger.debug("Wellness factor details saved to database")

        # Update user streaks if user_id provided
        if u_id:
            try:
                logger.debug("Updating streak data for user %s", u_id)
                current_date = _parse_current_date(json_input.get("local_date"))
                _update_user_streaks(u_id, current_date)
            except Exception as exc:
                logger.warning("⚠️ Streak update failed: %s", exc)
                logger.warning(
                    "Details: Prediction still saved to database, "
                    "only streak tracking failed."
                )

        db.session.commit()
        logger.info("💾 Database save completed for %s", prediction_id)


def read_from_db(prediction_id=None, user_id=None):
    """Read prediction data from database."""
    from flask import current_app

    if current_app.config.get("DB_DISABLED", False):
        return {"error": "Database disabled", "status": "disabled"}

    try:
        base_query = db.select(Predictions).options(
            selectinload(Predictions.details).selectinload(PredDetails.advices),
            selectinload(Predictions.details).selectinload(PredDetails.references),
        )

        if prediction_id:
            try:
                pred_uuid = uuid.UUID(prediction_id)
            except ValueError:
                return {
                    "error": (
                        "Invalid prediction_id format. Must be a valid UUID " "string."
                    ),
                    "status": "bad_request",
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
                    "status": "bad_request",
                }

            stmt = base_query.filter(Predictions.user_id == user_uuid).order_by(
                Predictions.pred_date.desc()
            )
            predictions = db.session.execute(stmt).scalars().all()

            if not predictions:
                return {
                    "error": "No predictions found for this user",
                    "status": "not_found",
                }
        else:
            return {
                "error": "Either prediction_id or user_id must be provided",
                "status": "bad_request",
            }

        # Format data
        result = []
        for pred in predictions:
            pred_data = {
                "prediction_id": str(pred.pred_id),
                "user_id": str(pred.user_id) if pred.user_id else None,
                "guest_id": str(pred.guest_id) if pred.guest_id else None,
                "prediction_date": (
                    pred.pred_date.isoformat() if pred.pred_date else None
                ),
                "input_data": {
                    "screen_time_hours": pred.screen_time,
                    "work_screen_hours": pred.work_screen,
                    "leisure_screen_hours": pred.leisure_screen,
                    "sleep_hours": pred.sleep_hours,
                    "sleep_quality_1_5": pred.sleep_quality,
                    "stress_level_0_10": pred.stress_level,
                    "productivity_0_100": pred.productivity,
                    "exercise_minutes_per_week": pred.exercise,
                    "social_hours_per_week": pred.social,
                },
                "prediction_score": pred.pred_score,
                "ai_desc": pred.ai_desc,
                "details": [],
            }

            for detail in pred.details:
                detail_data = {
                    "factor_name": detail.factor_name,
                    "impact_score": detail.impact_score,
                    "factor_type": (
                        detail.factor_type
                        if detail.factor_type is not None
                        else "improvement"
                    ),
                    "advices": [a.advice_text for a in detail.advices],
                    "references": [r.reference_link for r in detail.references],
                }
                pred_data["details"].append(detail_data)
            result.append(pred_data)

        if prediction_id:
            return {"status": "success", "data": result[0] if result else None}
        else:
            return {
                "status": "success",
                "data": result,
                "total_predictions": len(result),
            }

    except Exception as e:
        logger.error("Error reading from database: %s", e, exc_info=True)
        return {"error": str(e), "status": "error"}
