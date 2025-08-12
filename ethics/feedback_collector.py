import jax
import jax.numpy as jnp
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackCollector:
    def __init__(self):
        self.feedback_store = []

    def collect(self, input_text: str, output_text: str, feedback_score: float, metadata: Dict = None):
        """Store feedback for a given input-output pair."""
        if not 0 <= feedback_score <= 1:
            raise ValueError("Feedback score must be between 0 and 1")
        self.feedback_store.append({
            "input": input_text,
            "output": output_text,
            "feedback_score": feedback_score,
            "metadata": metadata or {}
        })
        logger.info(f"Collected feedback: score={feedback_score}, input={input_text[:50]}..., output={output_text[:50]}...")

    def get_feedback_dataset(self) -> List[Dict]:
        """Return collected feedback as a dataset."""
        return self.feedback_store

    def clear(self):
        """Clear stored feedback."""
        self.feedback_store = []
        logger.info("Feedback store cleared")