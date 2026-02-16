"""
Tests for Zero-Shot Reasoning Module

Tests for conceptual understanding, knowledge graph operations,
and reasoning chains without explicit training.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np


class TestConceptType(unittest.TestCase):
    """Test ConceptType enum."""
    
    def test_concept_types_exist(self):
        """Test all concept types are defined."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptType
        
        self.assertEqual(ConceptType.ABSTRACT.value, "abstract")
        self.assertEqual(ConceptType.CONCRETE.value, "concrete")
        self.assertEqual(ConceptType.RELATIONAL.value, "relational")
        self.assertEqual(ConceptType.TEMPORAL.value, "temporal")
        self.assertEqual(ConceptType.CAUSAL.value, "causal")
        self.assertEqual(ConceptType.MATHEMATICAL.value, "mathematical")


class TestConceptNode(unittest.TestCase):
    """Test ConceptNode dataclass."""
    
    def test_concept_node_creation(self):
        """Test creating concept nodes."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptNode, ConceptType
        
        embedding = jnp.zeros(64)
        
        node = ConceptNode(
            concept_id="concept_001",
            name="gravity",
            concept_type=ConceptType.CAUSAL,
            embedding=embedding,
            relations={"causes": {"falling"}, "part_of": {"physics"}},
            properties={"strength": "weak", "range": "infinite"},
            confidence=0.95
        )
        
        self.assertEqual(node.concept_id, "concept_001")
        self.assertEqual(node.name, "gravity")
        self.assertEqual(node.concept_type, ConceptType.CAUSAL)
        self.assertEqual(node.confidence, 0.95)
        self.assertEqual(node.usage_count, 0)
    
    def test_concept_node_defaults(self):
        """Test concept node default values."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptNode, ConceptType
        
        node = ConceptNode(
            concept_id="test",
            name="test_concept",
            concept_type=ConceptType.ABSTRACT,
            embedding=jnp.zeros(32),
            relations={},
            properties={},
            confidence=0.5
        )
        
        self.assertEqual(node.usage_count, 0)
        self.assertEqual(node.last_accessed, 0.0)


class TestReasoningChain(unittest.TestCase):
    """Test ReasoningChain dataclass."""
    
    def test_reasoning_chain_creation(self):
        """Test creating reasoning chains."""
        from src.modules.capabilities.zero_shot_reasoning import ReasoningChain
        
        chain = ReasoningChain(
            steps=[
                {"action": "observe", "result": "apple falls"},
                {"action": "infer", "result": "gravity pulls apple"}
            ],
            confidence_scores=[0.9, 0.85],
            final_conclusion="Gravity causes objects to fall",
            evidence=["Newton's laws", "observation"],
            reasoning_type="causal"
        )
        
        self.assertEqual(len(chain.steps), 2)
        self.assertEqual(len(chain.confidence_scores), 2)
        self.assertEqual(chain.reasoning_type, "causal")


class TestConceptualKnowledgeGraph(unittest.TestCase):
    """Test ConceptualKnowledgeGraph class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptualKnowledgeGraph
        self.graph = ConceptualKnowledgeGraph(embedding_dim=64)
    
    def test_graph_initialization(self):
        """Test knowledge graph initialization."""
        self.assertEqual(self.graph.embedding_dim, 64)
        self.assertEqual(len(self.graph.concepts), 0)
        self.assertIn("is_a", self.graph.relation_types)
        self.assertIn("causes", self.graph.relation_types)
    
    def test_add_concept(self):
        """Test adding concepts to the graph."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptNode, ConceptType
        
        concept = ConceptNode(
            concept_id="dog",
            name="dog",
            concept_type=ConceptType.CONCRETE,
            embedding=jnp.ones(64) * 0.5,
            relations={"is_a": {"animal"}},
            properties={"legs": 4},
            confidence=0.99
        )
        
        self.graph.add_concept(concept)
        
        self.assertIn("dog", self.graph.concepts)
        self.assertEqual(self.graph.concepts["dog"].name, "dog")
    
    def test_find_similar_concepts(self):
        """Test finding similar concepts."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptNode, ConceptType
        
        # Add some concepts
        for i, name in enumerate(["cat", "dog", "car", "tree"]):
            concept = ConceptNode(
                concept_id=name,
                name=name,
                concept_type=ConceptType.CONCRETE,
                embedding=jnp.array([float(i)] * 64),
                relations={},
                properties={},
                confidence=0.9
            )
            self.graph.add_concept(concept)
        
        # Find similar to cat
        query = jnp.zeros(64)  # Similar to cat's embedding
        similar = self.graph.find_similar_concepts(query, top_k=2)
        
        self.assertLessEqual(len(similar), 2)
    
    def test_find_similar_empty_graph(self):
        """Test finding similar concepts in empty graph."""
        query = jnp.zeros(64)
        similar = self.graph.find_similar_concepts(query)
        
        self.assertEqual(len(similar), 0)
    
    def test_multi_hop_reasoning(self):
        """Test multi-hop reasoning between concepts."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptNode, ConceptType
        
        # Create a concept chain: A -> B -> C
        concepts = [
            ConceptNode(
                concept_id="A",
                name="A",
                concept_type=ConceptType.ABSTRACT,
                embedding=jnp.zeros(64),
                relations={"causes": {"B"}},
                properties={},
                confidence=0.9
            ),
            ConceptNode(
                concept_id="B",
                name="B",
                concept_type=ConceptType.ABSTRACT,
                embedding=jnp.zeros(64),
                relations={"causes": {"C"}},
                properties={},
                confidence=0.9
            ),
            ConceptNode(
                concept_id="C",
                name="C",
                concept_type=ConceptType.ABSTRACT,
                embedding=jnp.zeros(64),
                relations={},
                properties={},
                confidence=0.9
            )
        ]
        
        for concept in concepts:
            self.graph.add_concept(concept)
        
        # Find paths from A to C
        paths = self.graph.multi_hop_reasoning("A", "C", max_hops=3)
        
        # Should find at least one path
        self.assertGreater(len(paths), 0)
    
    def test_multi_hop_caching(self):
        """Test that multi-hop results are cached."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptNode, ConceptType
        
        # Add concepts
        for name in ["X", "Y"]:
            concept = ConceptNode(
                concept_id=name,
                name=name,
                concept_type=ConceptType.ABSTRACT,
                embedding=jnp.zeros(64),
                relations={"causes": {"Y"}} if name == "X" else {},
                properties={},
                confidence=0.9
            )
            self.graph.add_concept(concept)
        
        # First call
        paths1 = self.graph.multi_hop_reasoning("X", "Y", max_hops=2)
        
        # Second call should use cache
        paths2 = self.graph.multi_hop_reasoning("X", "Y", max_hops=2)
        
        self.assertEqual(paths1, paths2)
        self.assertIn("X_Y_2", self.graph.multi_hop_cache)
    
    def test_get_concept_relations(self):
        """Test getting concept relations."""
        from src.modules.capabilities.zero_shot_reasoning import ConceptNode, ConceptType
        
        concept = ConceptNode(
            concept_id="test",
            name="test",
            concept_type=ConceptType.RELATIONAL,
            embedding=jnp.zeros(64),
            relations={
                "is_a": {"parent1", "parent2"},
                "causes": {"effect1"}
            },
            properties={},
            confidence=0.9
        )
        self.graph.add_concept(concept)
        
        # Get all relations
        all_relations = self.graph.get_concept_relations("test")
        self.assertIn("is_a", all_relations)
        self.assertIn("causes", all_relations)
        
        # Get filtered relations
        is_a_relations = self.graph.get_concept_relations("test", relation_type="is_a")
        self.assertIn("is_a", is_a_relations)


class TestZeroShotConceptualSystem(unittest.TestCase):
    """Test ZeroShotConceptualSystem class."""
    
    def test_system_initialization(self):
        """Test zero-shot system initialization."""
        try:
            from src.modules.capabilities.zero_shot_reasoning import ZeroShotConceptualSystem
            
            system = ZeroShotConceptualSystem(d_model=128)
            self.assertEqual(system.d_model, 128)
        except AttributeError:
            self.skipTest("ZeroShotConceptualSystem not available")


class TestReasoningIntegration(unittest.TestCase):
    """Integration tests for reasoning system."""
    
    def test_concept_to_reasoning_chain(self):
        """Test creating reasoning chains from concepts."""
        from src.modules.capabilities.zero_shot_reasoning import (
            ConceptualKnowledgeGraph, ConceptNode, ConceptType, ReasoningChain
        )
        
        graph = ConceptualKnowledgeGraph(embedding_dim=64)
        
        # Build a causal chain
        concepts = [
            ("rain", {"causes": {"wet_ground"}}),
            ("wet_ground", {"causes": {"slippery"}}),
            ("slippery", {"causes": {"accidents"}})
        ]
        
        for name, relations in concepts:
            concept = ConceptNode(
                concept_id=name,
                name=name,
                concept_type=ConceptType.CAUSAL,
                embedding=jnp.zeros(64),
                relations=relations,
                properties={},
                confidence=0.9
            )
            graph.add_concept(concept)
        
        # Reason from rain to accidents
        paths = graph.multi_hop_reasoning("rain", "accidents", max_hops=4)
        
        # Should find the causal chain
        self.assertGreater(len(paths), 0)


if __name__ == "__main__":
    unittest.main()
