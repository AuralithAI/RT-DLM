"""
Unit tests for advanced_understanding/comprehension_modules.py

Tests the following components:
- SemanticParser: Main semantic parsing module
- GraphAttentionLayer: GNN layer with attention
- ConceptualGraphBuilder: Graph construction from text
- KnowledgeExtractor: Knowledge extraction from graphs
- MultiHopReasoner: Multi-hop reasoning over graphs
- compute_graph_accuracy: Graph accuracy metrics
"""

import sys
import unittest
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from src.modules.capabilities.comprehension_modules import (
    SemanticParser,
    GraphAttentionLayer,
    ConceptualGraphBuilder,
    KnowledgeExtractor,
    MultiHopReasoner,
    ConceptualGraph,
    compute_graph_accuracy,
    create_semantic_parser_fn,
)


# Test constants
D_MODEL = 64
BATCH_SIZE = 2
SEQ_LEN = 16
MAX_NODES = 16
NUM_HEADS = 4
NUM_HOPS = 2


class TestGraphAttentionLayer(unittest.TestCase):
    """Test GraphAttentionLayer with multi-head attention"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(42)
        
    def test_initialization(self):
        """Test that GraphAttentionLayer initializes correctly"""
        def init_fn(node_features, adjacency):
            layer = GraphAttentionLayer(d_model=D_MODEL, num_heads=NUM_HEADS)
            return layer(node_features, adjacency, is_training=True)
        
        init = hk.transform(init_fn)
        node_features = jnp.ones((BATCH_SIZE, MAX_NODES, D_MODEL))
        adjacency = jnp.eye(MAX_NODES)[None, :, :].repeat(BATCH_SIZE, axis=0)
        params = init.init(self.rng, node_features, adjacency)
        self.assertIsNotNone(params)
        
    def test_output_shape(self):
        """Test output shapes are correct"""
        def forward_fn(node_features, adjacency):
            layer = GraphAttentionLayer(d_model=D_MODEL, num_heads=NUM_HEADS)
            return layer(node_features, adjacency, is_training=False)
        
        transformed = hk.transform(forward_fn)
        
        node_features = jnp.ones((BATCH_SIZE, MAX_NODES, D_MODEL))
        adjacency = jnp.eye(MAX_NODES)[None, :, :].repeat(BATCH_SIZE, axis=0)
        
        params = transformed.init(self.rng, node_features, adjacency)
        updated_features, attention_weights = transformed.apply(
            params, self.rng, node_features, adjacency
        )
        
        # Check output shapes
        self.assertEqual(updated_features.shape, (BATCH_SIZE, MAX_NODES, D_MODEL))
        self.assertEqual(attention_weights.shape, (BATCH_SIZE, NUM_HEADS, MAX_NODES, MAX_NODES))
        
    def test_attention_masking(self):
        """Test that attention respects adjacency mask"""
        def forward_fn(node_features, adjacency):
            layer = GraphAttentionLayer(d_model=D_MODEL, num_heads=NUM_HEADS, dropout_rate=0.0)
            return layer(node_features, adjacency, is_training=False)
        
        transformed = hk.transform(forward_fn)
        
        node_features = jax.random.normal(self.rng, (BATCH_SIZE, MAX_NODES, D_MODEL))
        # Sparse adjacency - only self-loops
        adjacency = jnp.eye(MAX_NODES)[None, :, :].repeat(BATCH_SIZE, axis=0)
        
        params = transformed.init(self.rng, node_features, adjacency)
        _, attention_weights = transformed.apply(
            params, self.rng, node_features, adjacency
        )
        
        # With only self-loops, attention should be concentrated on diagonal
        # After softmax, each node should have attention weight 1.0 on itself
        diagonal_attention = jnp.diagonal(attention_weights, axis1=-2, axis2=-1)
        
        # All diagonal elements should be close to 1.0 (only valid connection)
        self.assertTrue(jnp.allclose(diagonal_attention, 1.0, atol=1e-5))


class TestConceptualGraphBuilder(unittest.TestCase):
    """Test ConceptualGraphBuilder for graph construction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(43)
        
    def test_graph_building(self):
        """Test that graph builder creates valid graphs"""
        def build_fn(text_embeddings):
            builder = ConceptualGraphBuilder(
                d_model=D_MODEL, 
                max_nodes=MAX_NODES,
                edge_threshold=0.3
            )
            return builder(text_embeddings)
        
        transformed = hk.transform(build_fn)
        
        text_embeddings = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, text_embeddings)
        graph = transformed.apply(params, self.rng, text_embeddings)
        
        # Check graph structure
        self.assertEqual(graph.node_features.shape, (BATCH_SIZE, MAX_NODES, D_MODEL))
        self.assertEqual(graph.adjacency_matrix.shape, (BATCH_SIZE, MAX_NODES, MAX_NODES))
        
        # Adjacency should have self-loops (diagonal = 1)
        diagonal = jnp.diagonal(graph.adjacency_matrix, axis1=-2, axis2=-1)
        self.assertTrue(jnp.all(diagonal >= 1.0))
        
    def test_node_types(self):
        """Test that node type classification works"""
        def build_fn(text_embeddings):
            builder = ConceptualGraphBuilder(
                d_model=D_MODEL, 
                max_nodes=MAX_NODES
            )
            return builder(text_embeddings)
        
        transformed = hk.transform(build_fn)
        
        text_embeddings = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, text_embeddings)
        graph = transformed.apply(params, self.rng, text_embeddings)
        
        # Check node types are valid indices
        self.assertEqual(graph.node_types.shape, (BATCH_SIZE, MAX_NODES))
        self.assertTrue(jnp.all(graph.node_types >= 0))
        self.assertTrue(jnp.all(graph.node_types < 8))  # 8 node types


class TestSemanticParser(unittest.TestCase):
    """Test SemanticParser main module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(44)
        
    def test_full_parsing_pipeline(self):
        """Test full semantic parsing pipeline"""
        def parse_fn(text_input, query):
            parser = SemanticParser(
                d_model=D_MODEL,
                max_nodes=MAX_NODES,
                num_hops=NUM_HOPS,
                num_heads=NUM_HEADS
            )
            return parser.parse(text_input, query, is_training=False)
        
        transformed = hk.transform(parse_fn)
        
        text_input = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        query = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        
        params = transformed.init(self.rng, text_input, query)
        result = transformed.apply(params, self.rng, text_input, query)
        
        # Check result structure
        self.assertIn("graph", result)
        self.assertIn("knowledge", result)
        self.assertIn("reasoning", result)
        self.assertIn("semantic_representation", result)
        
        # Check semantic representation shape
        self.assertEqual(result["semantic_representation"].shape, (BATCH_SIZE, D_MODEL))
        
    def test_build_graph_method(self):
        """Test build_graph method creates valid conceptual graph"""
        def build_fn(text_input):
            parser = SemanticParser(d_model=D_MODEL, max_nodes=MAX_NODES)
            return parser.build_graph(text_input)
        
        transformed = hk.transform(build_fn)
        
        text_input = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, text_input)
        graph = transformed.apply(params, self.rng, text_input)
        
        # Verify graph is a ConceptualGraph with correct properties
        self.assertEqual(graph.node_features.shape, (BATCH_SIZE, MAX_NODES, D_MODEL))
        self.assertEqual(graph.adjacency_matrix.shape, (BATCH_SIZE, MAX_NODES, MAX_NODES))
        
    def test_parsing_without_query(self):
        """Test parsing works without query (no multi-hop reasoning)"""
        def parse_fn(text_input):
            parser = SemanticParser(d_model=D_MODEL, max_nodes=MAX_NODES)
            return parser.parse(text_input, query=None, is_training=False)
        
        transformed = hk.transform(parse_fn)
        
        text_input = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        
        params = transformed.init(self.rng, text_input)
        result = transformed.apply(params, self.rng, text_input)
        
        # Reasoning should be None when no query provided
        self.assertIsNone(result["reasoning"])
        # But graph and knowledge should still be present
        self.assertIsNotNone(result["graph"])
        self.assertIsNotNone(result["knowledge"])


class TestMultiHopReasoner(unittest.TestCase):
    """Test MultiHopReasoner for graph traversal"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(45)
        
    def test_multi_hop_traversal(self):
        """Test multi-hop reasoning over graph"""
        def reason_fn(query, node_features, adjacency):
            # Create a graph manually
            graph = ConceptualGraph(
                node_features=node_features,
                adjacency_matrix=adjacency
            )
            reasoner = MultiHopReasoner(
                d_model=D_MODEL,
                num_hops=NUM_HOPS,
                num_heads=NUM_HEADS
            )
            return reasoner(query, graph, is_training=False)
        
        transformed = hk.transform(reason_fn)
        
        query = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        node_features = jax.random.normal(self.rng, (BATCH_SIZE, MAX_NODES, D_MODEL))
        adjacency = jnp.eye(MAX_NODES)[None, :, :].repeat(BATCH_SIZE, axis=0)
        
        params = transformed.init(self.rng, query, node_features, adjacency)
        result = transformed.apply(params, self.rng, query, node_features, adjacency)
        
        # Check output structure
        self.assertIn("answer_embedding", result)
        self.assertIn("hop_embeddings", result)
        self.assertIn("reasoning_trace", result)
        
        # Check answer shape
        self.assertEqual(result["answer_embedding"].shape, (BATCH_SIZE, D_MODEL))
        
        # Check we have correct number of hops
        self.assertEqual(len(result["hop_embeddings"]), NUM_HOPS)


class TestGraphAccuracy(unittest.TestCase):
    """Test graph accuracy computation utilities"""
    
    def test_perfect_accuracy(self):
        """Test accuracy computation with perfect prediction"""
        predicted = jnp.array([[[1, 1, 0], [1, 1, 1], [0, 1, 1]]])
        target = jnp.array([[[1, 1, 0], [1, 1, 1], [0, 1, 1]]])
        
        metrics = compute_graph_accuracy(predicted, target)
        
        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=5)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=5)
        self.assertAlmostEqual(metrics["recall"], 1.0, places=5)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=5)
        
    def test_zero_accuracy(self):
        """Test accuracy computation with completely wrong prediction"""
        predicted = jnp.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        target = jnp.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        
        metrics = compute_graph_accuracy(predicted, target)
        
        # All predictions are false positives
        self.assertAlmostEqual(metrics["precision"], 0.0, places=5)
        self.assertAlmostEqual(metrics["recall"], 0.0, places=5)
        
    def test_partial_accuracy(self):
        """Test accuracy computation with partial match"""
        # 50% overlap
        predicted = jnp.array([[[1, 1], [0, 0]]])
        target = jnp.array([[[1, 0], [0, 1]]])
        
        metrics = compute_graph_accuracy(predicted, target)
        
        # 1 TP (top-left), 1 FP (top-right), 1 FN (bottom-right), 1 TN (bottom-left)
        # Accuracy = (1+1)/(1+1+1+1) = 0.5
        self.assertAlmostEqual(metrics["accuracy"], 0.5, places=5)


class TestKnowledgeExtractor(unittest.TestCase):
    """Test KnowledgeExtractor for structured knowledge"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(46)
        
    def test_knowledge_extraction(self):
        """Test knowledge extraction from graph"""
        def extract_fn(node_features, adjacency):
            graph = ConceptualGraph(
                node_features=node_features,
                adjacency_matrix=adjacency
            )
            extractor = KnowledgeExtractor(d_model=D_MODEL)
            return extractor(graph)
        
        transformed = hk.transform(extract_fn)
        
        node_features = jax.random.normal(self.rng, (BATCH_SIZE, MAX_NODES, D_MODEL))
        adjacency = jnp.eye(MAX_NODES)[None, :, :].repeat(BATCH_SIZE, axis=0)
        
        params = transformed.init(self.rng, node_features, adjacency)
        result = transformed.apply(params, self.rng, node_features, adjacency)
        
        # Check output structure
        self.assertIn("entity_embeddings", result)
        self.assertIn("entity_weights", result)
        self.assertIn("relation_probs", result)
        self.assertIn("triple_scores", result)
        self.assertIn("fact_representations", result)
        
        # Entity weights should sum to 1 (softmax)
        entity_weights = result["entity_weights"]
        weight_sums = entity_weights.sum(axis=-1)
        self.assertTrue(jnp.allclose(weight_sums, 1.0, atol=1e-5))


class TestSemanticParserFactory(unittest.TestCase):
    """Test create_semantic_parser_fn factory function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rng = jax.random.PRNGKey(47)
        
    def test_factory_creates_valid_transform(self):
        """Test factory creates valid Haiku transformed function"""
        parser_transform = create_semantic_parser_fn(
            d_model=D_MODEL,
            max_nodes=MAX_NODES,
            num_hops=NUM_HOPS
        )
        
        text_input = jax.random.normal(self.rng, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        query = jax.random.normal(self.rng, (BATCH_SIZE, D_MODEL))
        
        params = parser_transform.init(self.rng, text_input, query)
        result = parser_transform.apply(params, self.rng, text_input, query)
        
        self.assertIn("semantic_representation", result)
        self.assertEqual(result["semantic_representation"].shape, (BATCH_SIZE, D_MODEL))


if __name__ == "__main__":
    unittest.main()

