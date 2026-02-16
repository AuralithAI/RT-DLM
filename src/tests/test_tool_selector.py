"""
Tests for Tool Selector Module

Tests for tool selection for the Reasoning Language Model (RLM).
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk


class TestToolSelection(unittest.TestCase):
    """Test ToolSelection dataclass."""
    
    def test_tool_selection_creation(self):
        """Test creating ToolSelection."""
        from src.core.rlm.tool_selector import ToolSelection
        from src.config.rlm_config import ToolType
        
        selection = ToolSelection(
            tool=ToolType.PEEK,
            confidence=0.95,
            parameters={"start": 0, "length": 100},
            reasoning="Need to look at file content"
        )
        
        self.assertEqual(selection.tool, ToolType.PEEK)
        self.assertEqual(selection.confidence, 0.95)
        self.assertIn("start", selection.parameters)
    
    def test_tool_selection_without_reasoning(self):
        """Test ToolSelection without reasoning."""
        from src.core.rlm.tool_selector import ToolSelection
        from src.config.rlm_config import ToolType
        
        selection = ToolSelection(
            tool=ToolType.GREP,
            confidence=0.8,
            parameters={"pattern": "test"}
        )
        
        self.assertIsNone(selection.reasoning)


class TestToolSelector(unittest.TestCase):
    """Test ToolSelector class."""
    
    def test_selector_initialization(self):
        """Test ToolSelector initialization."""
        from src.core.rlm.tool_selector import ToolSelector
        
        def init_fn():
            selector = ToolSelector(d_model=64, num_tools=5)
            query = jnp.zeros((1, 64))
            context = jnp.zeros((1, 64))
            state = jnp.zeros((1, 64))
            return selector(query, context, state)
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        
        self.assertIsNotNone(params)
    
    def test_selector_forward_pass(self):
        """Test selector forward pass."""
        from src.core.rlm.tool_selector import ToolSelector
        
        def forward_fn(query, context, state):
            selector = ToolSelector(d_model=64, num_tools=5)
            return selector(query, context, state)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 64))
        context = jax.random.normal(rng, (2, 64))
        state = jax.random.normal(rng, (2, 64))
        
        params = init.init(rng, query, context, state)
        output = init.apply(params, rng, query, context, state)
        
        self.assertIsNotNone(output)
    
    def test_selector_with_tool_history(self):
        """Test selector with tool history."""
        from src.core.rlm.tool_selector import ToolSelector
        
        def forward_fn(query, context, state, history):
            selector = ToolSelector(d_model=64, num_tools=5)
            return selector(query, context, state, tool_history=history)
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 64))
        context = jax.random.normal(rng, (2, 64))
        state = jax.random.normal(rng, (2, 64))
        history = jax.random.normal(rng, (2, 5, 64))  # 5 previous tools
        
        params = init.init(rng, query, context, state, history)
        output = init.apply(params, rng, query, context, state, history)
        
        self.assertIsNotNone(output)


class TestQueryEncoder(unittest.TestCase):
    """Test query encoding."""
    
    def test_query_encoder_shape(self):
        """Test query encoder output shape."""
        from src.core.rlm.tool_selector import ToolSelector
        
        def forward_fn(query, context, state):
            selector = ToolSelector(d_model=64, num_tools=5)
            encoded_query = selector.query_encoder(query)
            return encoded_query
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 64))
        context = jax.random.normal(rng, (2, 64))
        state = jax.random.normal(rng, (2, 64))
        
        params = init.init(rng, query, context, state)
        encoded = init.apply(params, rng, query, context, state)
        
        # Output should have same d_model dimension
        self.assertEqual(encoded.shape[-1], 64)


class TestContextEncoder(unittest.TestCase):
    """Test context encoding."""
    
    def test_context_encoder_shape(self):
        """Test context encoder output shape."""
        from src.core.rlm.tool_selector import ToolSelector
        
        def forward_fn(query, context, state):
            selector = ToolSelector(d_model=64, num_tools=5)
            encoded_context = selector.context_encoder(context)
            return encoded_context
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 64))
        context = jax.random.normal(rng, (2, 64))
        state = jax.random.normal(rng, (2, 64))
        
        params = init.init(rng, query, context, state)
        encoded = init.apply(params, rng, query, context, state)
        
        self.assertEqual(encoded.shape[-1], 64)


class TestToolClassifier(unittest.TestCase):
    """Test tool classification."""
    
    def test_classifier_output_shape(self):
        """Test tool classifier output shape matches num_tools."""
        from src.core.rlm.tool_selector import ToolSelector
        
        num_tools = 5
        
        def forward_fn(query, context, state):
            selector = ToolSelector(d_model=64, num_tools=num_tools)
            # Combine query and context
            combined = jnp.concatenate([
                selector.query_encoder(query),
                selector.context_encoder(context)
            ], axis=-1)
            logits = selector.tool_classifier(combined)
            return logits
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 64))
        context = jax.random.normal(rng, (2, 64))
        state = jax.random.normal(rng, (2, 64))
        
        params = init.init(rng, query, context, state)
        logits = init.apply(params, rng, query, context, state)
        
        # Should output logits for each tool
        self.assertEqual(logits.shape[-1], num_tools)


class TestTerminationHead(unittest.TestCase):
    """Test termination head."""
    
    def test_termination_output_range(self):
        """Test termination head output is between 0 and 1."""
        from src.core.rlm.tool_selector import ToolSelector
        
        def forward_fn(query, context, state):
            selector = ToolSelector(d_model=64, num_tools=5)
            combined = jnp.concatenate([
                selector.query_encoder(query),
                selector.context_encoder(context)
            ], axis=-1)
            # Get first d_model features for termination
            termination = selector.termination_head(combined[..., :64])
            return termination
        
        init = hk.transform(forward_fn)
        rng = jax.random.PRNGKey(42)
        
        query = jax.random.normal(rng, (2, 64))
        context = jax.random.normal(rng, (2, 64))
        state = jax.random.normal(rng, (2, 64))
        
        params = init.init(rng, query, context, state)
        termination = init.apply(params, rng, query, context, state)
        
        # Termination probability should be between 0 and 1
        self.assertTrue(jnp.all(termination >= 0))
        self.assertTrue(jnp.all(termination <= 1))


class TestParameterHeads(unittest.TestCase):
    """Test parameter prediction heads."""
    
    def test_parameter_heads_exist(self):
        """Test parameter heads are initialized."""
        from src.core.rlm.tool_selector import ToolSelector
        
        def init_fn():
            selector = ToolSelector(d_model=64, num_tools=5)
            return list(selector.parameter_heads.keys())
        
        init = hk.transform(init_fn)
        rng = jax.random.PRNGKey(42)
        params = init.init(rng)
        heads = init.apply(params, rng)
        
        # Should have parameter heads for different tools
        self.assertIn("peek_start", heads)
        self.assertIn("peek_length", heads)


class TestToolType(unittest.TestCase):
    """Test ToolType enum."""
    
    def test_tool_types_exist(self):
        """Test ToolType enum values exist."""
        from src.config.rlm_config import ToolType
        
        # Check basic tool types exist
        self.assertIsNotNone(ToolType.PEEK)
        self.assertIsNotNone(ToolType.GREP)


class TestTemperatureScaling(unittest.TestCase):
    """Test temperature scaling in tool selection."""
    
    def test_temperature_affects_distribution(self):
        """Test that temperature affects softmax distribution."""
        logits = jnp.array([[1.0, 2.0, 3.0]])
        
        # Low temperature - more peaked
        probs_low = jax.nn.softmax(logits / 0.1)
        
        # High temperature - more uniform
        probs_high = jax.nn.softmax(logits / 1.0)
        
        # Low temperature should have higher max probability
        self.assertGreater(jnp.max(probs_low), jnp.max(probs_high))


if __name__ == "__main__":
    unittest.main()
