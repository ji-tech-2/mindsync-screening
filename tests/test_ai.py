"""
Unit tests for ai.py - AI advice generation functions
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock


class TestGetAIAdvice:
    """Test get_ai_advice function."""
    
    @patch('flaskr.ai.genai.Client')
    def test_successful_advice_generation(self, mock_client_class):
        """Test successful AI advice generation."""
        from flaskr.ai import get_ai_advice
        
        # Mock response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "description": "Test advice description",
            "factors": {
                "Sleep Quality": {
                    "advices": ["Tip 1", "Tip 2", "Tip 3"],
                    "references": [
                        {"title": "Sleep Guide", "url": "https://example.com"}
                    ]
                }
            }
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        wellness_analysis = {
            'areas_for_improvement': [
                {'feature': 'Sleep Quality', 'impact_score': 2.5}
            ]
        }
        
        result = get_ai_advice(75, "Good", wellness_analysis, "test_api_key")
        
        assert "description" in result
        assert "factors" in result
        assert result["description"] == "Test advice description"
    
    @patch('flaskr.ai.genai.Client')
    def test_handles_json_decode_error(self, mock_client_class):
        """Test handling of JSON decode error."""
        from flaskr.ai import get_ai_advice
        
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.text = "Invalid JSON"
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        wellness_analysis = {'areas_for_improvement': []}
        
        result = get_ai_advice(75, "Good", wellness_analysis, "test_api_key")
        
        assert "description" in result
        assert "factors" in result
        assert "issue" in result["description"].lower()
    
    @patch('flaskr.ai.genai.Client')
    def test_handles_api_exception(self, mock_client_class):
        """Test handling of API exception."""
        from flaskr.ai import get_ai_advice
        
        mock_client_class.side_effect = Exception("API Error")
        
        wellness_analysis = {'areas_for_improvement': []}
        
        result = get_ai_advice(75, "Good", wellness_analysis, "test_api_key")
        
        assert "description" in result
        assert "factors" in result
        assert "issue" in result["description"].lower()
    
    def test_handles_empty_wellness_analysis(self):
        """Test handling of empty wellness analysis."""
        from flaskr.ai import get_ai_advice
        
        with patch('flaskr.ai.genai.Client') as mock_client_class:
            mock_response = Mock()
            mock_response.text = json.dumps({
                "description": "General advice",
                "factors": {}
            })
            
            mock_client = Mock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client
            
            # Empty wellness analysis
            result = get_ai_advice(75, "Good", None, "test_api_key")
            
            assert "description" in result
            assert "factors" in result


class TestGetWeeklyAdvice:
    """Test get_weekly_advice function."""
    
    @patch('flaskr.ai.genai.Client')
    def test_successful_weekly_advice(self, mock_client_class):
        """Test successful weekly advice generation."""
        from flaskr.ai import get_weekly_advice
        
        # Mock response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "description": "Weekly summary",
            "factors": {
                "Sleep Quality": {
                    "advices": ["Weekly tip 1", "Weekly tip 2", "Weekly tip 3"],
                    "references": []
                }
            }
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        top_factors = [
            {"factor_name": "Sleep Quality", "count": 5, "avg_impact_score": 2.5}
        ]
        
        result = get_weekly_advice(top_factors, "test_api_key")
        
        assert "description" in result
        assert "factors" in result
        assert "Weekly summary" in result["description"]
    
    def test_handles_empty_factors(self):
        """Test handling empty factors list."""
        from flaskr.ai import get_weekly_advice
        
        result = get_weekly_advice([], "test_api_key")
        
        assert "description" in result
        assert "factors" in result
        assert "No critical factors" in result["description"] or "Great job" in result["description"]
    
    @patch('flaskr.ai.genai.Client')
    def test_handles_api_exception(self, mock_client_class):
        """Test handling of API exception."""
        from flaskr.ai import get_weekly_advice
        
        mock_client_class.side_effect = Exception("API Error")
        
        top_factors = [
            {"factor_name": "Sleep Quality", "count": 3, "avg_impact_score": 1.8}
        ]
        
        result = get_weekly_advice(top_factors, "test_api_key")
        
        assert "description" in result
        assert "factors" in result
        assert "issue" in result["description"].lower()


class TestGetDailyAdvice:
    """Test get_daily_advice function."""
    
    @patch('flaskr.ai.genai.Client')
    def test_successful_daily_advice(self, mock_client_class):
        """Test successful daily advice generation."""
        from flaskr.ai import get_daily_advice
        
        # Mock response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "Sleep Quality": "Get 7-8 hours tonight",
            "Exercise": "Take a 20-minute walk"
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        top_factors = [
            {"factor_name": "Sleep Quality", "impact_score": 2.5},
            {"factor_name": "Exercise", "impact_score": 1.8}
        ]
        
        result = get_daily_advice(top_factors, "test_api_key")
        
        assert isinstance(result, dict)
        assert "Sleep Quality" in result
        assert "Exercise" in result
    
    def test_handles_empty_factors(self):
        """Test handling empty factors list."""
        from flaskr.ai import get_daily_advice
        
        result = get_daily_advice([], "test_api_key")
        
        assert isinstance(result, str)
        assert "great" in result.lower() or "good" in result.lower()
    
    @patch('flaskr.ai.genai.Client')
    def test_handles_api_exception(self, mock_client_class):
        """Test handling of API exception."""
        from flaskr.ai import get_daily_advice
        
        mock_client_class.side_effect = Exception("API Error")
        
        top_factors = [
            {"factor_name": "Sleep Quality", "impact_score": 2.5}
        ]
        
        result = get_daily_advice(top_factors, "test_api_key")
        
        assert isinstance(result, dict)
        assert "error" in result or len(result) > 0


class TestAIIntegration:
    """Integration tests for AI functionality."""
    
    @patch('flaskr.ai.genai.Client')
    def test_consistent_response_structure(self, mock_client_class):
        """Test that AI responses have consistent structure."""
        from flaskr.ai import get_ai_advice, get_weekly_advice
        
        # Mock consistent responses
        advice_response = Mock()
        advice_response.text = json.dumps({
            "description": "Test description",
            "factors": {
                "Sleep": {
                    "advices": ["Advice 1", "Advice 2", "Advice 3"],
                    "references": [{"title": "Test", "url": "https://test.com"}]
                }
            }
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = advice_response
        mock_client_class.return_value = mock_client
        
        wellness_analysis = {
            'areas_for_improvement': [
                {'feature': 'Sleep', 'impact_score': 2.0}
            ]
        }
        
        # Test get_ai_advice
        result1 = get_ai_advice(70, "Good", wellness_analysis, "test_key")
        assert "description" in result1
        assert "factors" in result1
        
        # Test get_weekly_advice
        top_factors = [{"factor_name": "Sleep", "count": 5, "avg_impact_score": 2.0}]
        result2 = get_weekly_advice(top_factors, "test_key")
        assert "description" in result2
        assert "factors" in result2
    
    @patch('flaskr.ai.genai.Client')
    def test_handles_multiple_factors(self, mock_client_class):
        """Test handling of multiple wellness factors."""
        from flaskr.ai import get_ai_advice
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "description": "Multi-factor advice",
            "factors": {
                "Sleep": {"advices": ["Tip 1", "Tip 2", "Tip 3"], "references": []},
                "Exercise": {"advices": ["Tip 1", "Tip 2", "Tip 3"], "references": []},
                "Stress": {"advices": ["Tip 1", "Tip 2", "Tip 3"], "references": []}
            }
        })
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        wellness_analysis = {
            'areas_for_improvement': [
                {'feature': 'Sleep', 'impact_score': 2.5},
                {'feature': 'Exercise', 'impact_score': 2.0},
                {'feature': 'Stress', 'impact_score': 1.8}
            ]
        }
        
        result = get_ai_advice(65, "Fair", wellness_analysis, "test_key")
        
        assert len(result["factors"]) == 3
        assert "Sleep" in result["factors"]
        assert "Exercise" in result["factors"]
        assert "Stress" in result["factors"]
