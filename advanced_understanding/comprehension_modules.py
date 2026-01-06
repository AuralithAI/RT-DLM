"""
Advanced Semantic Parsing and Comprehension Modules

This module implements graph-based semantic parsing for advanced comprehension
and knowledge extraction. Uses GNN layers with Haiku Linear + attention to 
build conceptual graphs from inputs.

Key Components:
- SemanticParser: Main module for building conceptual graphs from text
- GraphAttentionLayer: GNN layer with multi-head attention for message passing
- ConceptualGraphBuilder: Builds adjacency matrices and node representations
- KnowledgeExtractor: Extracts structured knowledge from conceptual graphs
- MultiHopReasoner: Performs multi-hop reasoning over knowledge graphs

Why:
- Strengthens zero-shot reasoning for niches like health diagnostics
- Enables better abstraction from multimodal data
- Supports multi-hop reasoning for complex queries

References:
- Graph Attention Networks (Veličković et al., 2018)
- Semantic Role Labeling with Graph Attention Networks
- Knowledge Graph Embeddings for Reasoning
"""

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ConceptualGraph:
    """Represents a conceptual graph extracted from input"""
    node_features: jnp.ndarray  # [batch, num_nodes, d_model]
    adjacency_matrix: jnp.ndarray  # [batch, num_nodes, num_nodes]
    edge_features: Optional[jnp.ndarray] = None  # [batch, num_nodes, num_nodes, edge_dim]
    node_types: Optional[jnp.ndarray] = None  # [batch, num_nodes] - node type indices
    edge_types: Optional[jnp.ndarray] = None  # [batch, num_nodes, num_nodes] - edge type indices
    attention_weights: Optional[jnp.ndarray] = None  # [batch, num_heads, num_nodes, num_nodes]


class GraphAttentionLayer(hk.Module):
    """
    Graph Attention Network (GAT) layer with multi-head attention.
    
    Performs message passing with attention-weighted aggregation of 
    neighbor features. Uses Haiku Linear layers for transformations.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout rate for attention weights
        name: Module name
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        
    def __call__(
        self, 
        node_features: jnp.ndarray,
        adjacency_matrix: jnp.ndarray,
        is_training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply graph attention layer.
        
        Args:
            node_features: [batch, num_nodes, d_model]
            adjacency_matrix: [batch, num_nodes, num_nodes] - binary or weighted
            is_training: Whether in training mode (for dropout)
            
        Returns:
            updated_features: [batch, num_nodes, d_model]
            attention_weights: [batch, num_heads, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Linear projections for Q, K, V
        query_proj = hk.Linear(self.d_model, name="query_projection")
        key_proj = hk.Linear(self.d_model, name="key_projection")
        value_proj = hk.Linear(self.d_model, name="value_projection")
        
        # Project and reshape for multi-head attention
        queries = query_proj(node_features)
        keys = key_proj(node_features)
        values = value_proj(node_features)
        
        # Reshape: [batch, num_nodes, num_heads, head_dim]
        queries = queries.reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, num_heads, num_nodes, head_dim]
        queries = jnp.transpose(queries, (0, 2, 1, 3))
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = jnp.transpose(values, (0, 2, 1, 3))
        
        # Compute attention scores: [batch, num_heads, num_nodes, num_nodes]
        attention_scores = jnp.einsum('bhid,bhjd->bhij', queries, keys)
        attention_scores = attention_scores / jnp.sqrt(self.head_dim)
        
        # Apply graph structure mask (only attend to connected nodes)
        # Expand adjacency for heads: [batch, 1, num_nodes, num_nodes]
        adj_mask = adjacency_matrix[:, None, :, :]
        
        # Mask non-connected nodes with large negative value
        attention_scores = jnp.where(
            adj_mask > 0,
            attention_scores,
            jnp.full_like(attention_scores, -1e9)
        )
        
        # Softmax over neighbors
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # Apply dropout during training
        if is_training and self.dropout_rate > 0:
            dropout_mask = hk.dropout(
                hk.next_rng_key(),
                self.dropout_rate,
                jnp.ones_like(attention_weights)
            )
            attention_weights = attention_weights * dropout_mask
        
        # Aggregate neighbor features: [batch, num_heads, num_nodes, head_dim]
        aggregated = jnp.einsum('bhij,bhjd->bhid', attention_weights, values)
        
        # Reshape back: [batch, num_nodes, d_model]
        aggregated = jnp.transpose(aggregated, (0, 2, 1, 3))
        aggregated = aggregated.reshape(batch_size, num_nodes, self.d_model)
        
        # Output projection
        output_proj = hk.Linear(self.d_model, name="output_projection")
        updated_features = output_proj(aggregated)
        
        # Residual connection and layer norm
        layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="layer_norm")
        updated_features = layer_norm(node_features + updated_features)
        
        return updated_features, attention_weights


class ConceptualGraphBuilder(hk.Module):
    """
    Builds conceptual graphs from text embeddings.
    
    Creates adjacency matrices using attention-based edge prediction
    and extracts node features representing semantic concepts.
    
    Args:
        d_model: Model dimension
        max_nodes: Maximum number of nodes in the graph
        edge_threshold: Threshold for edge creation
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int,
        max_nodes: int = 32,
        edge_threshold: float = 0.3,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.edge_threshold = edge_threshold
        
    def __call__(
        self,
        text_embeddings: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> ConceptualGraph:
        """
        Build conceptual graph from text embeddings.
        
        Args:
            text_embeddings: [batch, seq_len, d_model]
            mask: Optional mask for valid positions [batch, seq_len]
            
        Returns:
            ConceptualGraph containing node features and adjacency matrix
        """
        batch_size, seq_len, _ = text_embeddings.shape
        
        # Concept extraction: compress sequence to key concepts
        concept_extractor = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(self.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="concept_extractor")
        
        # Extract concept representations
        concepts = concept_extractor(text_embeddings)
        
        # Pool to max_nodes using attention-based selection
        if seq_len > self.max_nodes:
            # Importance scoring for node selection
            importance_scorer = hk.Linear(1, name="importance_scorer")
            importance_scores = importance_scorer(concepts).squeeze(-1)  # [batch, seq_len]
            
            if mask is not None:
                importance_scores = jnp.where(mask > 0, importance_scores, -1e9)
            
            # Select top-k most important nodes
            # Cast to jnp.ndarray for type checker compatibility
            scores_array: jnp.ndarray = jnp.asarray(importance_scores)
            _, top_indices = jax.lax.top_k(scores_array, self.max_nodes)
            
            # Gather selected nodes
            batch_indices = jnp.arange(batch_size)[:, None]
            node_features = concepts[batch_indices, top_indices]  # [batch, max_nodes, d_model]
        else:
            # Pad to max_nodes
            pad_size = self.max_nodes - seq_len
            node_features = jnp.pad(
                concepts, 
                ((0, 0), (0, pad_size), (0, 0)),
                mode='constant',
                constant_values=0
            )
        
        # Build adjacency matrix using edge prediction
        adjacency_matrix = self._build_adjacency(node_features)
        
        # Compute node types (entity classification)
        node_type_classifier = hk.Linear(8, name="node_type_classifier")  # 8 node types
        node_type_logits = node_type_classifier(node_features)
        node_types = jnp.argmax(node_type_logits, axis=-1)
        
        # Compute edge types using bilinear scoring
        edge_types = self._classify_edges(node_features, adjacency_matrix)
        
        return ConceptualGraph(
            node_features=node_features,
            adjacency_matrix=adjacency_matrix,
            node_types=node_types,
            edge_types=edge_types
        )
        
    def _build_adjacency(self, node_features: jnp.ndarray) -> jnp.ndarray:
        """
        Build adjacency matrix using learned edge prediction.
        
        Args:
            node_features: [batch, num_nodes, d_model]
            
        Returns:
            adjacency_matrix: [batch, num_nodes, num_nodes]
        """
        # Edge predictor using bilinear attention
        edge_head = hk.Linear(self.d_model // 2, name="edge_head")
        edge_tail = hk.Linear(self.d_model // 2, name="edge_tail")
        
        heads = edge_head(node_features)  # [batch, num_nodes, d_model//2]
        tails = edge_tail(node_features)  # [batch, num_nodes, d_model//2]
        
        # Compute edge scores: [batch, num_nodes, num_nodes]
        edge_scores = jnp.einsum('bid,bjd->bij', heads, tails)
        edge_scores = edge_scores / jnp.sqrt(self.d_model // 2)
        
        # Apply sigmoid and threshold for binary adjacency
        edge_probs = jax.nn.sigmoid(edge_scores)
        adjacency_matrix = jnp.where(edge_probs > self.edge_threshold, 1.0, 0.0)
        
        # Add self-loops
        eye = jnp.eye(self.max_nodes)[None, :, :]
        adjacency_matrix = jnp.maximum(adjacency_matrix, eye)
        
        return adjacency_matrix
        
    def _classify_edges(
        self, 
        node_features: jnp.ndarray,
        adjacency_matrix: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Classify edge types for existing edges.
        
        Args:
            node_features: [batch, num_nodes, d_model]
            adjacency_matrix: [batch, num_nodes, num_nodes]
            
        Returns:
            edge_types: [batch, num_nodes, num_nodes] - edge type indices
        """
        num_edge_types = 6  # e.g., causes, relates_to, is_a, part_of, contradicts, supports
        
        # Edge type classifier
        edge_type_scorer = hk.Linear(num_edge_types, name="edge_type_scorer")
        
        # Compute edge representations
        # Expand for pairwise: [batch, num_nodes, 1, d_model] and [batch, 1, num_nodes, d_model]
        heads_expanded = node_features[:, :, None, :]
        tails_expanded = node_features[:, None, :, :]
        
        # Concatenate head and tail features
        edge_features = heads_expanded + tails_expanded  # Simple additive composition
        
        # Classify edges
        edge_logits = edge_type_scorer(edge_features)  # [batch, num_nodes, num_nodes, num_edge_types]
        edge_types = jnp.argmax(edge_logits, axis=-1)  # [batch, num_nodes, num_nodes]
        
        # Mask non-edges
        edge_types = jnp.where(adjacency_matrix > 0, edge_types, -1)
        
        return edge_types


class KnowledgeExtractor(hk.Module):
    """
    Extracts structured knowledge from conceptual graphs.
    
    Identifies key entities, relationships, and facts from the
    graph structure for downstream reasoning.
    
    Args:
        d_model: Model dimension
        num_relation_types: Number of relation types to extract
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int,
        num_relation_types: int = 16,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_relation_types = num_relation_types
        
    def __call__(
        self,
        graph: ConceptualGraph
    ) -> Dict[str, jnp.ndarray]:
        """
        Extract structured knowledge from conceptual graph.
        
        Args:
            graph: ConceptualGraph with node features and adjacency
            
        Returns:
            Dictionary containing:
            - entity_embeddings: Key entity representations
            - relation_embeddings: Relation representations
            - triple_scores: Confidence scores for (head, relation, tail) triples
            - fact_representations: Aggregated fact embeddings
        """
        node_features = graph.node_features
        adjacency_matrix = graph.adjacency_matrix
        
        _, num_nodes, _ = node_features.shape
        
        # Entity importance scoring
        entity_scorer = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(1)
        ], name="entity_scorer")
        
        entity_scores = entity_scorer(node_features).squeeze(-1)  # [batch, num_nodes]
        entity_weights = jax.nn.softmax(entity_scores, axis=-1)  # [batch, num_nodes]
        
        # Weighted entity embeddings
        entity_embeddings = node_features * entity_weights[:, :, None]
        
        # Relation extraction using edge features
        relation_extractor = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(self.num_relation_types)
        ], name="relation_extractor")
        
        # Compute pairwise relation scores
        heads_expanded = node_features[:, :, None, :]  # [batch, num_nodes, 1, d_model]
        tails_expanded = node_features[:, None, :, :]  # [batch, 1, num_nodes, d_model]
        
        # Relation representation (concatenation)
        edge_representation = heads_expanded * tails_expanded  # Element-wise product
        relation_logits = relation_extractor(edge_representation)  # [batch, n, n, num_relations]
        relation_probs = jax.nn.softmax(relation_logits, axis=-1)
        
        # Triple scoring (head, relation, tail)
        triple_scorer = hk.Linear(1, name="triple_scorer")
        triple_input = jnp.concatenate([
            edge_representation, 
            relation_probs
        ], axis=-1)
        triple_scores = jax.nn.sigmoid(triple_scorer(triple_input).squeeze(-1))
        
        # Mask based on adjacency
        triple_scores = triple_scores * adjacency_matrix
        
        # Aggregate facts
        fact_aggregator = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(self.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="fact_aggregator")
        
        # Weighted sum of edge representations
        weighted_edges = edge_representation * triple_scores[:, :, :, None]
        fact_representations = fact_aggregator(
            weighted_edges.sum(axis=(1, 2)) / (num_nodes ** 2)
        )
        
        return {
            "entity_embeddings": entity_embeddings,
            "entity_weights": entity_weights,
            "relation_probs": relation_probs,
            "triple_scores": triple_scores,
            "fact_representations": fact_representations
        }


class MultiHopReasoner(hk.Module):
    """
    Performs multi-hop reasoning over knowledge graphs.
    
    Uses iterative message passing and attention to traverse
    the graph for answering complex queries.
    
    Args:
        d_model: Model dimension
        num_hops: Maximum number of reasoning hops
        num_heads: Number of attention heads
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int,
        num_hops: int = 3,
        num_heads: int = 4,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_hops = num_hops
        self.num_heads = num_heads
        
    def __call__(
        self,
        query: jnp.ndarray,
        graph: ConceptualGraph,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning from query over graph.
        
        Args:
            query: Query embedding [batch, d_model]
            graph: ConceptualGraph for reasoning
            is_training: Whether in training mode
            
        Returns:
            Dictionary containing:
            - answer_embedding: Final answer representation
            - hop_embeddings: Intermediate hop representations
            - path_attention: Attention weights at each hop
            - reasoning_trace: Trace of reasoning path
        """
        node_features = graph.node_features
        adjacency_matrix = graph.adjacency_matrix
        
        hop_embeddings = []
        path_attentions = []
        reasoning_trace = []
        
        # Initialize reasoning state with query
        query_proj = hk.Linear(self.d_model, name="query_projection")
        current_state = query_proj(query)  # [batch, d_model]
        
        for hop in range(self.num_hops):
            # Create GAT layer for this hop
            gat_layer = GraphAttentionLayer(
                self.d_model, 
                num_heads=self.num_heads,
                name=f"gat_hop_{hop}"
            )
            
            # Update node features with current state context
            state_expanded = current_state[:, None, :]  # [batch, 1, d_model]
            state_broadcast = jnp.broadcast_to(state_expanded, node_features.shape)
            
            # Combine nodes with state via gating
            gate_proj = hk.Linear(self.d_model, name=f"gate_hop_{hop}")
            gate = jax.nn.sigmoid(gate_proj(jnp.concatenate([node_features, state_broadcast], axis=-1)))
            gated_features = node_features * gate
            
            # Apply GAT
            updated_nodes, attention_weights = gat_layer(
                gated_features, adjacency_matrix, is_training
            )
            
            # Query-guided attention to select relevant nodes
            query_attention = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.d_model // self.num_heads,
                name=f"query_attention_hop_{hop}",
                w_init=hk.initializers.TruncatedNormal(stddev=0.02)
            )
            
            # Reshape query for attention: [batch, 1, d_model]
            query_for_attention = current_state[:, None, :]
            attended_nodes = query_attention(query_for_attention, updated_nodes, updated_nodes)
            
            # Update reasoning state
            hop_embedding = attended_nodes.squeeze(1)  # [batch, d_model]
            
            state_update = hk.Sequential([
                hk.Linear(self.d_model * 2),
                jax.nn.silu,
                hk.Linear(self.d_model),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ], name=f"state_update_hop_{hop}")
            
            current_state = state_update(
                jnp.concatenate([current_state, hop_embedding], axis=-1)
            )
            
            hop_embeddings.append(hop_embedding)
            path_attentions.append(attention_weights)
            reasoning_trace.append({
                "hop": hop,
                "state": current_state,
                "attention": attention_weights
            })
        
        # Final answer synthesis
        answer_synthesizer = hk.Sequential([
            hk.Linear(self.d_model * 2),
            jax.nn.silu,
            hk.Linear(self.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="answer_synthesizer")
        
        # Combine all hop embeddings
        all_hops = jnp.stack(hop_embeddings, axis=1)  # [batch, num_hops, d_model]
        hop_summary = all_hops.mean(axis=1)  # [batch, d_model]
        
        answer_embedding = answer_synthesizer(
            jnp.concatenate([current_state, hop_summary], axis=-1)
        )
        
        return {
            "answer_embedding": answer_embedding,
            "hop_embeddings": hop_embeddings,
            "path_attentions": path_attentions,
            "reasoning_trace": reasoning_trace,
            "final_state": current_state
        }


class SemanticParser(hk.Module):
    """
    Main semantic parser module for advanced comprehension.
    
    Combines graph building, knowledge extraction, and multi-hop
    reasoning for semantic understanding and zero-shot reasoning.
    
    Designed for niches like health diagnostics where abstraction
    from multimodal data is critical.
    
    Args:
        d_model: Model dimension
        max_nodes: Maximum nodes in conceptual graph
        num_hops: Number of reasoning hops
        num_heads: Number of attention heads
        edge_threshold: Threshold for edge creation
        name: Module name
    """
    
    def __init__(
        self,
        d_model: int = 512,
        max_nodes: int = 32,
        num_hops: int = 3,
        num_heads: int = 8,
        edge_threshold: float = 0.3,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.edge_threshold = edge_threshold
        
    def build_graph(
        self,
        text_input: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> ConceptualGraph:
        """
        Build conceptual graph from text input.
        
        Args:
            text_input: Text embeddings [batch, seq_len, d_model]
            mask: Optional mask for valid positions [batch, seq_len]
            
        Returns:
            ConceptualGraph with node features and adjacency matrix
        """
        graph_builder = ConceptualGraphBuilder(
            d_model=self.d_model,
            max_nodes=self.max_nodes,
            edge_threshold=self.edge_threshold,
            name="graph_builder"
        )
        return graph_builder(text_input, mask)
        
    def extract_knowledge(
        self,
        graph: ConceptualGraph
    ) -> Dict[str, jnp.ndarray]:
        """
        Extract structured knowledge from conceptual graph.
        
        Args:
            graph: ConceptualGraph from build_graph
            
        Returns:
            Dictionary with entity embeddings, relations, and facts
        """
        extractor = KnowledgeExtractor(
            d_model=self.d_model,
            name="knowledge_extractor"
        )
        return extractor(graph)
        
    def reason_multi_hop(
        self,
        query: jnp.ndarray,
        graph: ConceptualGraph,
        is_training: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """
        Perform multi-hop reasoning over graph.
        
        Args:
            query: Query embedding [batch, d_model]
            graph: ConceptualGraph for reasoning
            is_training: Whether in training mode
            
        Returns:
            Dictionary with answer embedding and reasoning trace
        """
        reasoner = MultiHopReasoner(
            d_model=self.d_model,
            num_hops=self.num_hops,
            num_heads=self.num_heads,
            name="multi_hop_reasoner"
        )
        return reasoner(query, graph, is_training)
        
    def parse(
        self,
        text_input: jnp.ndarray,
        query: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Full semantic parsing pipeline.
        
        Args:
            text_input: Text embeddings [batch, seq_len, d_model]
            query: Optional query for reasoning [batch, d_model]
            mask: Optional mask for valid positions [batch, seq_len]
            is_training: Whether in training mode
            
        Returns:
            Dictionary containing:
            - graph: ConceptualGraph
            - knowledge: Extracted knowledge
            - reasoning: Multi-hop reasoning results (if query provided)
            - semantic_representation: Final semantic embedding
        """
        # Build conceptual graph
        graph = self.build_graph(text_input, mask)
        
        # Apply GNN layers for enriched node representations
        gat_layer1 = GraphAttentionLayer(
            self.d_model, 
            num_heads=self.num_heads,
            name="gat_layer_1"
        )
        gat_layer2 = GraphAttentionLayer(
            self.d_model,
            num_heads=self.num_heads,
            name="gat_layer_2"
        )
        
        # Message passing through GNN
        enriched_nodes, attention1 = gat_layer1(
            graph.node_features, graph.adjacency_matrix, is_training
        )
        enriched_nodes, attention2 = gat_layer2(
            enriched_nodes, graph.adjacency_matrix, is_training
        )
        
        # Update graph with enriched features
        enriched_graph = ConceptualGraph(
            node_features=enriched_nodes,
            adjacency_matrix=graph.adjacency_matrix,
            edge_features=graph.edge_features,
            node_types=graph.node_types,
            edge_types=graph.edge_types,
            attention_weights=attention2
        )
        
        # Extract knowledge
        knowledge = self.extract_knowledge(enriched_graph)
        
        # Multi-hop reasoning if query provided
        reasoning_result = None
        if query is not None:
            reasoning_result = self.reason_multi_hop(query, enriched_graph, is_training)
        
        # Compute final semantic representation
        semantic_projector = hk.Sequential([
            hk.Linear(self.d_model),
            jax.nn.silu,
            hk.Linear(self.d_model),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ], name="semantic_projector")
        
        # Aggregate node features for semantic embedding
        node_weights = knowledge["entity_weights"]  # [batch, num_nodes]
        weighted_nodes = enriched_nodes * node_weights[:, :, None]
        aggregated = weighted_nodes.sum(axis=1)  # [batch, d_model]
        
        semantic_representation = semantic_projector(aggregated)
        
        return {
            "graph": enriched_graph,
            "knowledge": knowledge,
            "reasoning": reasoning_result,
            "semantic_representation": semantic_representation,
            "node_attention_weights": [attention1, attention2]
        }
        
    def __call__(
        self,
        text_input: jnp.ndarray,
        query: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        is_training: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass: full semantic parsing.
        
        Args:
            text_input: Text embeddings [batch, seq_len, d_model]
            query: Optional query for reasoning [batch, d_model]
            mask: Optional mask for valid positions [batch, seq_len]
            is_training: Whether in training mode
            
        Returns:
            Semantic parsing results
        """
        return self.parse(text_input, query, mask, is_training)


# Utility functions for integration

def create_semantic_parser_fn(
    d_model: int = 512,
    max_nodes: int = 32,
    num_hops: int = 3,
    num_heads: int = 8,
    edge_threshold: float = 0.3
):
    """
    Create a transformed semantic parser function.
    
    Returns a Haiku transformed pair (init, apply) for the SemanticParser.
    """
    def _forward(text_input, query=None, mask=None, is_training=True):
        parser = SemanticParser(
            d_model=d_model,
            max_nodes=max_nodes,
            num_hops=num_hops,
            num_heads=num_heads,
            edge_threshold=edge_threshold
        )
        return parser(text_input, query, mask, is_training)
    
    return hk.transform(_forward)


def compute_graph_accuracy(
    predicted_adjacency: jnp.ndarray,
    target_adjacency: jnp.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute accuracy metrics for graph prediction.
    
    Args:
        predicted_adjacency: Predicted adjacency matrix [batch, n, n]
        target_adjacency: Ground truth adjacency [batch, n, n]
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    # Binarize predictions
    pred_binary = (predicted_adjacency > threshold).astype(jnp.float32)
    target_binary = (target_adjacency > 0).astype(jnp.float32)
    
    # True positives, false positives, false negatives
    tp = jnp.sum(pred_binary * target_binary)
    fp = jnp.sum(pred_binary * (1 - target_binary))
    fn = jnp.sum((1 - pred_binary) * target_binary)
    tn = jnp.sum((1 - pred_binary) * (1 - target_binary))
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def compute_knowledge_extraction_loss(
    knowledge_output: Dict[str, jnp.ndarray],
    target_entities: jnp.ndarray,
    target_relations: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute loss for knowledge extraction training.
    
    Args:
        knowledge_output: Output from KnowledgeExtractor
        target_entities: Ground truth entity labels [batch, num_nodes]
        target_relations: Ground truth relation labels [batch, n, n]
        
    Returns:
        Scalar loss value
    """
    # Entity importance loss (cross-entropy)
    entity_weights = knowledge_output["entity_weights"]
    entity_loss = -jnp.sum(target_entities * jnp.log(entity_weights + 1e-8))
    
    # Relation loss
    relation_probs = knowledge_output["relation_probs"]
    num_relations = relation_probs.shape[-1]
    target_one_hot = jax.nn.one_hot(target_relations, num_relations)
    relation_loss = -jnp.sum(target_one_hot * jnp.log(relation_probs + 1e-8))
    
    # Combined loss
    total_loss = entity_loss + relation_loss
    
    return total_loss
