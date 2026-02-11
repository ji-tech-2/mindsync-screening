"""
Unit tests for predict.py helper functions
"""

import uuid
from flaskr.predict import format_db_output


class TestFormatDBOutput:
    """Test format_db_output helper function."""

    def test_format_with_complete_data(self):
        """Test formatting with complete database output."""
        input_data = {
            "prediction_id": str(uuid.uuid4()),
            "prediction_score": 75.5,
            "prediction_date": "2026-02-08T10:30:00",
            "details": [
                {
                    "factor_name": "Sleep Quality",
                    "impact_score": 2.5,
                    "factor_type": "improvement",
                    "advices": ["Advice 1", "Advice 2"],
                    "references": ["https://example.com/sleep"],
                },
                {
                    "factor_name": "Exercise",
                    "impact_score": -1.5,
                    "factor_type": "strengths",
                    "advices": ["Keep it up"],
                    "references": [],
                },
            ],
            "ai_desc": "You're doing well overall.",
        }

        result = format_db_output(input_data)

        assert "wellness_analysis" in result
        assert "advice" in result
        assert "areas_for_improvement" in result["wellness_analysis"]
        assert "strengths" in result["wellness_analysis"]
        assert "description" in result["advice"]

    def test_format_with_missing_optional_fields(self):
        """Test formatting when optional fields are missing."""
        input_data = {
            "prediction_id": str(uuid.uuid4()),
            "prediction_score": 80.0,
            "prediction_date": "2026-02-08T10:30:00",
            "details": [],
        }

        result = format_db_output(input_data)

        assert "wellness_analysis" in result
        assert "advice" in result
        assert len(result["wellness_analysis"]["areas_for_improvement"]) == 0
        assert len(result["wellness_analysis"]["strengths"]) == 0

    def test_format_preserves_factor_structure(self):
        """Test that factor structure is preserved correctly."""
        input_data = {
            "prediction_id": str(uuid.uuid4()),
            "prediction_score": 70.0,
            "prediction_date": "2026-02-08T10:30:00",
            "details": [
                {
                    "factor_name": "Sleep Hours",
                    "impact_score": 3.2,
                    "factor_type": "improvement",
                    "advices": ["Sleep more", "Go to bed earlier"],
                    "references": ["https://sleep.org"],
                }
            ],
            "ai_desc": "Focus on sleep",
        }

        result = format_db_output(input_data)

        factors = result["wellness_analysis"]["areas_for_improvement"]
        assert len(factors) == 1
        assert factors[0]["feature"] == "Sleep Hours"
        assert factors[0]["impact_score"] == 3.2

    def test_format_creates_factors_dict_in_advice(self):
        """Test that factors dictionary is created in advice section."""
        input_data = {
            "prediction_id": str(uuid.uuid4()),
            "prediction_score": 65.0,
            "prediction_date": "2026-02-08T10:30:00",
            "details": [
                {
                    "factor_name": "Stress Level",
                    "impact_score": 2.8,
                    "factor_type": "improvement",
                    "advices": ["Meditate", "Take breaks"],
                    "references": ["https://stress.org"],
                },
                {
                    "factor_name": "Social Activity",
                    "impact_score": -2.0,
                    "factor_type": "strengths",
                    "advices": ["Maintain connections"],
                    "references": [],
                },
            ],
            "ai_desc": "Balance stress and social life",
        }

        result = format_db_output(input_data)

        assert "factors" in result["advice"]
        factors_dict = result["advice"]["factors"]

        # Only improvement factors should be in advice.factors
        assert "Stress Level" in factors_dict

        # Check factor structure
        stress_factor = factors_dict["Stress Level"]
        assert "advices" in stress_factor
        assert "references" in stress_factor
        assert len(stress_factor["advices"]) == 2


class TestPredictEndpointHelpers:
    """Test helper functions used by predict endpoints."""

    def test_uuid_validation(self):
        """Test UUID validation helper."""
        from flaskr.db import is_valid_uuid

        # Valid UUIDs
        assert is_valid_uuid(str(uuid.uuid4())) is True
        assert is_valid_uuid("550e8400-e29b-41d4-a716-446655440000") is True

        # Invalid UUIDs
        assert is_valid_uuid("not-a-uuid") is False
        assert is_valid_uuid("123") is False
        assert is_valid_uuid("") is False
        assert is_valid_uuid(None) is False


class TestDatabaseHelpers:
    """Test database helper functions."""

    def test_is_valid_uuid_with_uuid_object(self):
        """Test is_valid_uuid with UUID object."""
        from flaskr.db import is_valid_uuid

        test_uuid = uuid.uuid4()
        assert is_valid_uuid(test_uuid) is True

    def test_is_valid_uuid_with_string(self):
        """Test is_valid_uuid with string."""
        from flaskr.db import is_valid_uuid

        test_uuid_str = str(uuid.uuid4())
        assert is_valid_uuid(test_uuid_str) is True

    def test_is_valid_uuid_with_invalid_format(self):
        """Test is_valid_uuid with invalid formats."""
        from flaskr.db import is_valid_uuid

        assert is_valid_uuid("12345") is False
        assert is_valid_uuid("not-uuid-format") is False
        assert is_valid_uuid("550e8400-e29b-41d4-a716") is False  # Incomplete


class TestModelIntegrationWithPredict:
    """Test integration between model and predict modules."""

    def test_categorize_score_integration(self):
        """Test that categorization works for various scores."""
        from flaskr.model import categorize_mental_health_score

        # Test various score ranges
        scores_and_expected = [
            (10, "dangerous"),
            (20, "not healthy"),
            (40, "average"),
            (70, "healthy"),
        ]

        for score, expected_category in scores_and_expected:
            result = categorize_mental_health_score(score)
            assert (
                result == expected_category
            ), f"Score {score} should be {expected_category}"
