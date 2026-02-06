"""
Advanced Understanding Module

This module provides semantic parsing and advanced comprehension capabilities
using graph-based knowledge extraction with GNN layers.

Key Components:
- SemanticParser: Main module for building conceptual graphs from text
- GraphAttentionLayer: GNN layer with multi-head attention
- ConceptualGraphBuilder: Builds adjacency matrices and node representations
- KnowledgeExtractor: Extracts structured knowledge from graphs
- MultiHopReasoner: Multi-hop reasoning over knowledge graphs

Utility Functions:
- create_semantic_parser_fn: Factory for transformed SemanticParser
- compute_graph_accuracy: Metrics for graph prediction evaluation
- compute_knowledge_extraction_loss: Training loss for knowledge extraction
"""

from .comprehension_modules import (
    # Main module
    SemanticParser,
    
    # Core components
    GraphAttentionLayer,
    ConceptualGraphBuilder,
    KnowledgeExtractor,
    MultiHopReasoner,
    
    # Utility functions
    create_semantic_parser_fn,
    compute_graph_accuracy,
    compute_knowledge_extraction_loss,
)

__all__ = [
    # Main module
    "SemanticParser",
    
    # Core components
    "GraphAttentionLayer",
    "ConceptualGraphBuilder",
    "KnowledgeExtractor",
    "MultiHopReasoner",
    
    # Utility functions
    "create_semantic_parser_fn",
    "compute_graph_accuracy",
    "compute_knowledge_extraction_loss",
]

