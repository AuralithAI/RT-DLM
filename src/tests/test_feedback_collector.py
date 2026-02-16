"""
Tests for Feedback Collector Module

Tests for feedback collection functionality.
"""

import unittest


class TestFeedbackCollector(unittest.TestCase):
    """Test FeedbackCollector class."""
    
    def test_collector_initialization(self):
        """Test FeedbackCollector initialization."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        self.assertIsNotNone(collector)
        self.assertEqual(len(collector.feedback_store), 0)
    
    def test_collect_feedback(self):
        """Test collecting feedback."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        collector.collect(
            input_text="What is the capital of France?",
            output_text="The capital of France is Paris.",
            feedback_score=0.9
        )
        
        self.assertEqual(len(collector.feedback_store), 1)
        self.assertEqual(collector.feedback_store[0]["feedback_score"], 0.9)
    
    def test_collect_with_metadata(self):
        """Test collecting feedback with metadata."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        collector.collect(
            input_text="Test input",
            output_text="Test output",
            feedback_score=0.8,
            metadata={"user_id": "123", "session": "abc"}
        )
        
        self.assertEqual(len(collector.feedback_store), 1)
        self.assertIn("user_id", collector.feedback_store[0]["metadata"])
    
    def test_feedback_score_validation_lower_bound(self):
        """Test feedback score must be >= 0."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        with self.assertRaises(ValueError):
            collector.collect(
                input_text="Test",
                output_text="Test",
                feedback_score=-0.1
            )
    
    def test_feedback_score_validation_upper_bound(self):
        """Test feedback score must be <= 1."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        with self.assertRaises(ValueError):
            collector.collect(
                input_text="Test",
                output_text="Test",
                feedback_score=1.1
            )
    
    def test_valid_feedback_score_boundaries(self):
        """Test valid feedback scores at boundaries."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        # Score of 0 should be valid
        collector.collect(
            input_text="Test",
            output_text="Test",
            feedback_score=0.0
        )
        
        # Score of 1 should be valid
        collector.collect(
            input_text="Test",
            output_text="Test",
            feedback_score=1.0
        )
        
        self.assertEqual(len(collector.feedback_store), 2)


class TestGetFeedbackDataset(unittest.TestCase):
    """Test get_feedback_dataset method."""
    
    def test_get_empty_dataset(self):
        """Test getting empty dataset."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        dataset = collector.get_feedback_dataset()
        
        self.assertEqual(len(dataset), 0)
    
    def test_get_populated_dataset(self):
        """Test getting populated dataset."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        # Add multiple feedback entries
        for i in range(5):
            collector.collect(
                input_text=f"Input {i}",
                output_text=f"Output {i}",
                feedback_score=i / 5
            )
        
        dataset = collector.get_feedback_dataset()
        
        self.assertEqual(len(dataset), 5)
    
    def test_dataset_structure(self):
        """Test dataset entry structure."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        collector.collect(
            input_text="Test input",
            output_text="Test output",
            feedback_score=0.75
        )
        
        dataset = collector.get_feedback_dataset()
        entry = dataset[0]
        
        self.assertIn("input", entry)
        self.assertIn("output", entry)
        self.assertIn("feedback_score", entry)
        self.assertIn("metadata", entry)


class TestClear(unittest.TestCase):
    """Test clear method."""
    
    def test_clear_feedback_store(self):
        """Test clearing feedback store."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        # Add some feedback
        for i in range(3):
            collector.collect(
                input_text=f"Input {i}",
                output_text=f"Output {i}",
                feedback_score=0.5
            )
        
        self.assertEqual(len(collector.feedback_store), 3)
        
        # Clear
        collector.clear()
        
        self.assertEqual(len(collector.feedback_store), 0)
    
    def test_clear_allows_new_collection(self):
        """Test that clear allows new collection."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        collector.collect(
            input_text="Old",
            output_text="Data",
            feedback_score=0.5
        )
        
        collector.clear()
        
        collector.collect(
            input_text="New",
            output_text="Data",
            feedback_score=0.9
        )
        
        dataset = collector.get_feedback_dataset()
        
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["input"], "New")


class TestMetadataHandling(unittest.TestCase):
    """Test metadata handling."""
    
    def test_default_empty_metadata(self):
        """Test default empty metadata when not provided."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        collector.collect(
            input_text="Test",
            output_text="Test",
            feedback_score=0.5
        )
        
        self.assertEqual(collector.feedback_store[0]["metadata"], {})
    
    def test_complex_metadata(self):
        """Test complex metadata storage."""
        from src.core.ethics.feedback_collector import FeedbackCollector
        
        collector = FeedbackCollector()
        
        metadata = {
            "user_id": "user123",
            "timestamp": "2024-01-01T00:00:00",
            "tags": ["test", "experiment"],
            "nested": {"key": "value"}
        }
        
        collector.collect(
            input_text="Test",
            output_text="Test",
            feedback_score=0.7,
            metadata=metadata
        )
        
        stored_metadata = collector.feedback_store[0]["metadata"]
        
        self.assertEqual(stored_metadata["user_id"], "user123")
        self.assertEqual(stored_metadata["tags"], ["test", "experiment"])


if __name__ == "__main__":
    unittest.main()
