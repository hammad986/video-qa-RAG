"""
Test to verify the query rewrite module behaves correctly.
Ensures Gemini API is called for short queries and skipped for long queries.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from video_qa.query_rewriter import rewrite_query
from video_qa.config import config

class TestQueryRewriter(unittest.TestCase):
    
    @patch('video_qa.query_rewriter.genai.GenerativeModel')
    def test_short_query_rewrites(self, mock_model_class):
        # Mock Gemini response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "What is the main topic of python?"
        mock_response.prompt_feedback.block_reason = None
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        # Must temporarily ensure API key is present for test
        original_key = config.get("answer.gemini_api_key", "")
        config._config["answer"]["gemini_api_key"] = "test-key"
        
        result = rewrite_query("python")
        
        # Restore key
        config._config["answer"]["gemini_api_key"] = original_key
        
        self.assertEqual(result, "What is the main topic of python?")
        mock_model.generate_content.assert_called_once()
        
    @patch('video_qa.query_rewriter.genai.GenerativeModel')
    def test_long_query_skipped(self, mock_model_class):
        original = "What did the speaker say about the specific topic of regularization and overfitting?"
        # 13 words
        
        result = rewrite_query(original)
        
        self.assertEqual(result, original)
        mock_model_class.assert_not_called()

    @patch('video_qa.query_rewriter.genai.GenerativeModel')
    def test_api_failure_fallback(self, mock_model_class):
        # Mock API failure
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model
        
        original_key = config.get("answer.gemini_api_key", "")
        config._config["answer"]["gemini_api_key"] = "test-key"
        
        result = rewrite_query("regularization")
        
        config._config["answer"]["gemini_api_key"] = original_key
        
        # Should fallback to original without crashing
        self.assertEqual(result, "regularization")
        
if __name__ == '__main__':
    unittest.main()
