# RT-DLM Architecture

This document describes the model architecture for training.

> **Note**: Data collection, tokenization, and processing are handled by [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).

## Compute Controller

RT-DLM features a **learned Compute Controller** that dynamically allocates compute across modules under a budget constraint. Instead of running all modules on every input, the Controller decides which modules to invoke, how much budget to allocate, and when to halt—enabling adaptive, efficient inference.

### Controller Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    COMPUTE CONTROLLER                                               │
│                                  (core/agi/compute_controller.py)                                   │
│                                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                     ComputePlan                                               │  │
│  │                                   (K-Step Execution Loop)                                     │  │
│  │                                                                                               │  │
│  │   Input ──► ComputeState ──► Controller ──► ComputeAction ──► Module ──► Update State ──►... │  │
│  │             (hidden_state,    (learned      (module_probs,   (selected   (delta, confidence,  │  │
│  │              uncertainty,      policy)       budget_alloc,    module)     should_halt)        │  │
│  │              confidence,                     halt_prob)                                       │  │
│  │              budget_remaining)                                                                │  │
│  │                                                                                               │  │
│  │   Halting Conditions:                                                                         │  │
│  │   • halt_prob > halt_threshold (learned)                                                      │  │
│  │   • confidence > confidence_threshold (task complete)                                         │  │
│  │   • budget_remaining ≤ 0 (resource exhausted)                                                 │  │
│  │   • max_steps reached (safety limit)                                                          │  │
│  └───────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                     │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────────────────────────┐   │
│  │       ComputeController         │  │                  ModuleRegistry                        │   │
│  │       (Learned Policy)          │  │             (Available Modules)                        │   │
│  │                                 │  │                                                        │   │
│  │ Inputs:                         │  │  ┌──────────────────┐  ┌──────────────────┐            │   │
│  │ • Hidden state (from modules)   │  │  │ MEMORY_RETRIEVAL │  │ GRAPH_REASONING  │            │   │
│  │ • Uncertainty estimate          │  │  │ cost: 0.05-0.20  │  │ cost: 0.10-0.40  │            │   │
│  │ • Current confidence            │  │  └──────────────────┘  └──────────────────┘            │   │
│  │ • Budget remaining              │  │  ┌──────────────────┐  ┌──────────────────┐            │   │
│  │ • Memory summary                │  │  │ SYMBOLIC_REASON  │  │ PROBABILISTIC    │            │   │
│  │                                 │  │  │ cost: 0.05-0.25  │  │ cost: 0.08-0.30  │            │   │
│  │ Outputs:                        │  │  └──────────────────┘  └──────────────────┘            │   │
│  │ • module_probs (which to run)   │  │  ┌──────────────────┐  ┌──────────────────┐            │   │
│  │ • budget_allocation (per mod)   │  │  │ MOE_ROUTING      │  │ CONSCIOUSNESS    │            │   │
│  │ • halt_prob (when to stop)      │  │  │ cost: 0.15-0.50  │  │ cost: 0.20-0.60  │            │   │
│  │                                 │  │  └──────────────────┘  └──────────────────┘            │   │
│  │ Network:                        │  │  ┌──────────────────┐                                  │   │
│  │ • 2-layer MLP encoder           │  │  │ OUTPUT_GEN       │                                  │   │
│  │ • GRU state update              │  │  │ cost: 0.10-0.35  │                                  │   │
│  │ • Linear heads for outputs      │  │  └──────────────────┘                                  │   │
│  └─────────────────────────────────┘  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Module Contracts

Each module registered with the Controller implements a **ModuleContract** that specifies:

| Contract Field | Description |
|----------------|-------------|
| `base_cost` | Minimum compute cost (0.0-1.0) |
| `max_cost` | Maximum compute cost (0.0-1.0) |
| `capabilities` | What the module provides (e.g., "memory_retrieval", "logical_inference") |
| `dependencies` | Other modules that must run first |

### Module Output

When invoked, each module returns a **ModuleOutput**:

| Output Field | Description |
|--------------|-------------|
| `delta` | Update to hidden state |
| `confidence` | How confident the module is in its output |
| `actual_cost` | Compute actually consumed |
| `evidence` | Supporting information for the update |
| `should_halt` | Module's recommendation to stop |

### Configuration Presets

| Preset | Budget | Max Steps | Halt Threshold | Use Case |
|--------|--------|-----------|----------------|----------|
| `FAST_CONFIG` | 0.5 | 5 | 0.3 | Low-latency, simple queries |
| `BALANCED_CONFIG` | 1.0 | 10 | 0.5 | Default, general purpose |
| `THOROUGH_CONFIG` | 2.0 | 20 | 0.7 | Complex reasoning, quality-critical |

### Data Flow with Controller

```
                                           INPUT
                                             │
                                             ▼
                              ┌───────────────────────┐
                              │   ComputeController   │
                              │   (Learned Policy)    │
                              └───────────┬───────────┘
                                          │
               ┌──────────────────────────┼──────────────────────────┐
               │                          │                          │
               ▼                          ▼                          ▼
     ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
     │ Step 1: Memory   │      │ Step 2: Graph    │      │ Step 3: MoE      │
     │ (if selected)    │      │ (if selected)    │      │ (if selected)    │
     │ budget: 0.15     │      │ budget: 0.25     │      │ budget: 0.30     │
     └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  Controller       │
                              │  Halt Decision    │
                              │  (confidence=0.92)│
                              └─────────┬─────────┘
                                        │
                                        ▼
                                     OUTPUT
                        (after 3 steps, budget used: 0.70)
```

### Module Adapters

The `core/agi/module_adapters.py` provides adapters that wrap existing RT-DLM modules to comply with the `ModuleContract` interface:

| Adapter | Wraps | Capabilities |
|---------|-------|--------------|
| `MemoryRetrievalAdapter` | MemoryBank | memory_retrieval, context_augmentation |
| `GraphReasoningAdapter` | MultiHopGraphReasoner | relational_reasoning, graph_inference |
| `SymbolicReasoningAdapter` | SymbolicReasoning | logical_inference, rule_application |
| `ProbabilisticInferenceAdapter` | Probabilistic | uncertainty_estimation, bayesian_inference |
| `MoERoutingAdapter` | SparseMoE | expert_routing, specialized_processing |
| `ConsciousnessModuleAdapter` | ConsciousnessSimulator | metacognition, self_awareness |
| `OutputGenerationAdapter` | TransformerModel | text_generation, sequence_modeling |

### Training Losses

The `ControllerLossComputer` provides multi-objective training for the controller:

| Loss Component | Description | Weight |
|----------------|-------------|--------|
| `efficiency_loss` | Penalizes unnecessary computation; scales with task difficulty | λ_compute |
| `utilization_loss` | Encourages diverse module usage without overuse | λ_utilization |
| `calibration_loss` | Aligns confidence with actual accuracy (ECE-style) | λ_calibration |
| `budget_loss` | Penalizes overspending and excessive underspending | λ_budget |
| `ponder_loss` | Regularizes thinking time (PonderNet-style KL from geometric prior) | λ_ponder |

**Total Loss**: `task_loss + efficiency + utilization + calibration + budget + ponder`

### RL Reward Shaping

The `ControllerRewardShaper` provides dense training signals:

- **Step Rewards**: Reward confidence increases, penalize high uncertainty
- **Final Reward**: +1.0 for correct (with efficiency bonus), -0.5 for incorrect
- **Discounted Returns**: γ=0.99 for proper credit assignment

### AGI Integration

The `ControlledAGIForward` module replaces static module execution with controller-driven execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                  ControlledAGIForward                           │
│                                                                 │
│  Input ──► ControllerIntegrationMixin.create_module_executors   │
│                        │                                        │
│                        ▼                                        │
│              ComputeController (learned policy)                 │
│                        │                                        │
│                        ▼                                        │
│              ComputePlan (K-step loop)                          │
│                        │                                        │
│                        ▼                                        │
│              Output (with execution trace)                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key Classes**:
- `ControllerIntegrationMixin`: Creates module executors from AGI system components
- `ControlledAGIForward`: Haiku module for controller-driven forward pass
- `create_controlled_agi_fn`: Factory for transformed forward function

**Configuration Options**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_compute_controller` | `False` | Enable/disable controller |
| `controller_max_steps` | `10` | Maximum execution steps |
| `controller_initial_budget` | `1.0` | Starting compute budget |
| `controller_halt_threshold` | `0.8` | Confidence threshold for halting |
| `controller_confidence_threshold` | `0.9` | High-confidence early exit |
| `controller_strategy` | `"balanced"` | Preset: "fast", "balanced", "thorough" |
| `controller_lambda_compute` | `0.01` | Efficiency loss weight |
| `controller_lambda_utilization` | `0.005` | Utilization loss weight |
| `controller_lambda_calibration` | `0.1` | Calibration loss weight |
| `controller_lambda_budget` | `0.05` | Budget adherence loss weight |
| `controller_lambda_ponder` | `0.01` | Ponder cost loss weight |

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        RT-DLM MODEL                                                 │
│                                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                    COGNITIVE CORE                                             │  │
│  │                                                                                               │  │
│  │   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────────────┐   │  │
│  │   │ ConsciousnessSimulator │  │ ScientificDiscovery │  │     CreativeGenerationEngine     │   │  │
│  │   │                     │  │      Engine         │  │                                     │   │  │
│  │   │ - Self-awareness    │  │ - Hypothesis gen    │  │ - Novel content creation            │   │  │
│  │   │ - Introspection     │  │ - Experiment design │  │ - Multi-domain creativity           │   │  │
│  │   │ - Goal setting      │  │ - Knowledge synth   │  │ - Style transfer                    │   │  │
│  │   │ - Metacognition     │  │ - Scientific method │  │                                     │   │  │
│  │   └─────────────────────┘  └─────────────────────┘  └─────────────────────────────────────┘   │  │
│  │                                                                                               │  │
│  │   ┌─────────────────────────────────────────┐  ┌─────────────────────────────────────────┐    │  │
│  │   │     SocialEmotionalIntelligence         │  │          RTDLMAGISystem                 │    │  │
│  │   │                                         │  │         (Orchestrator)                  │    │  │
│  │   │ - Affective computing                   │  │                                         │    │  │
│  │   │ - Social context understanding          │  │ - Component coordination                │    │  │
│  │   │ - Empathy modeling                      │  │ - Task routing                          │    │  │
│  │   │ - Cultural awareness                    │  │ - Response synthesis                    │    │  │
│  │   └─────────────────────────────────────────┘  └─────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
┌─────────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
│         MODEL CORE              │ │     HYBRID ARCHITECTURE     │ │       QUANTUM LAYER         │
│       (core/model/)             │ │ (modules/hybrid_architecture)│ │      (core/quantum/)        │
│                                 │ │                             │ │                             │
│  ┌─────────────────────────┐    │ │  ┌─────────────────────┐    │ │  ┌─────────────────────┐    │
│  │      TMSModel           │    │ │  │ HybridArchitecture  │    │ │  │  QuantumSimulator   │    │
│  │                         │    │ │  │    Integrator       │    │ │  │                     │    │
│  │ - SelfAttentionModel    │    │ │  │                     │    │ │  │ - PHASE gate        │    │
│  │   (Advanced Attention)  │    │ │  │ - TraditionalML     │    │ │  │ - CNOT gate         │    │
│  │ - TransformerModel      │    │ │  │   - SVMLikeClassifier│   │ │  │ - Hadamard gate     │    │
│  │ - SparseMoE             │    │ │  │   - RandomForestLike│    │ │  │ - Rotation gates    │    │
│  │ - EthicalRewardModel    │    │ │  │   - NaiveBayesLike  │    │ │  │ - 100+ qubits (TN)  │    │
│  └─────────────────────────┘    │ │  │                     │    │ │  └─────────────────────┘    │
│                                 │ │  │ - DeepLearning      │    │ │                             │
│  ┌─────────────────────────┐    │ │  │   - CNNBranch       │    │ │  ┌─────────────────────┐    │
│  │   Advanced Attention    │    │ │  │   - RNNBranch       │    │ │  │ VariationalQuantum  │    │
│  │  (advanced_attention.py)│    │ │  │   - TransformerBranch│   │ │  │     Circuit         │    │
│  │                         │    │ │  │                     │    │ │  │                     │    │
│  │ - RoPE (Rotary Pos Emb) │    │ │  │ - SymbolicReasoning │    │ │  │ - build_layers()    │    │
│  │ - GQA (Grouped-Query)   │    │ │  │ - Probabilistic     │    │ │  │ - Parameterized     │    │
│  │ - MQA (Multi-Query)     │    │ │  └─────────────────────┘    │ │  │ - Trainable params  │    │
│  │ - Sliding Window        │    │ │                             │ │  └─────────────────────┘    │
│  │ - Linear Attention      │    │ │  ┌─────────────────────┐    │ │                             │
│  │ - Spiking Attention     │    │ │  │   EnsembleFusion    │    │ │  ┌─────────────────────┐    │
│  └─────────────────────────┘    │ │  │                     │    │ │  │ QuantumInspired     │    │
│                                 │ │  │ - Cross-paradigm    │    │ │  │    Attention        │    │
│  ┌─────────────────────────┐    │ │  │   interaction       │    │ │  │                     │    │
│  │      MemoryBank         │    │ │  │ - Outer-product     │    │ │  │ - Superposition     │    │
│  │                         │    │ │  │   attention         │    │ │  │ - Entanglement      │    │
│  │ - LTM (Long-Term)       │    │ │  │ - Adaptive fusion   │    │ │  │ - Quantum gates     │    │
│  │ - STM (Short-Term)      │    │ │  └─────────────────────┘    │ │  └─────────────────────┘    │
│  │ - MTM (Meta-Task)       │    │ │                             │ │                             │
│  │ - FAISS indexing        │    │ │  ┌─────────────────────┐    │ │  ┌─────────────────────┐    │
│  │ - Adaptive forgetting   │    │ │  │  MultiAgentConsensus│    │ │  │QuantumNeuralNetwork │    │
│  └─────────────────────────┘    │ │  │                     │    │ │  │                     │    │
│                                 │ │  │ - SpecialistAgents  │    │ │  │ - QuantumEncoder    │    │
│  ┌─────────────────────────┐    │ │  │   - Reasoning       │    │ │  │ - QuantumDecoder    │    │
│  │      SparseMoE          │    │ │  │   - Creativity      │    │ │  │ - QuantumAttention  │    │
│  │                         │    │ │  │   - Analysis        │    │ │  └─────────────────────┘    │
│  │ - AdaptiveGatingNetwork │    │ │  │   - Synthesis       │    │ └─────────────────────────────┘
│  │ - Top-K expert routing  │    │ │  │ - Weighted voting   │    │
│  │ - Load balancing loss   │    │ │  │ - Consensus loop    │    │
│  │ - Specialization loss   │    │ │  └─────────────────────┘    │
│  │ - Dynamic capacity      │    │ │                             │
│  │ - Router jitter         │    │ │  ┌─────────────────────┐    │
│  └─────────────────────────┘    │ │  │   Graph Neurons     │    │
│                                 │ │  │  (graph_neurons.py) │    │
│  ┌─────────────────────────┐    │ │  │                     │    │
│  │   Speculative Decoding  │    │ │  │ - GraphNeuron       │    │
│  │    (core/sampling.py)   │    │ │  │ - GraphAttentionUnit│    │
│  │                         │    │ │  │ - MultiHopReasoner  │    │
│  │ - SpeculativeDecoder    │    │ │  │ - GraphMoE          │    │
│  │ - SelfSpeculativeDecoder│    │ │  └─────────────────────┘    │
│  │ - Draft/verify pipeline │    │ │                             │
│  └─────────────────────────┘    │ └─────────────────────────────┘
└─────────────────────────────────┘
                                    │  └─────────────────────┘    │
                                    └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      MULTIMODAL PROCESSING                                          │
│                                     (modules/multimodal/)                                           │
│                                                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │      MultiModalRTDLM        │  │     HybridAudioEncoder      │  │     HybridVideoEncoder      │  │
│  │      (fusion_module.py)     │  │   (hybrid_audio_module.py)  │  │   (hybrid_video_module.py)  │  │
│  │                             │  │                             │  │                             │  │
│  │ - CrossModalAttention       │  │ - SignalProcessingBackbone  │  │ - HybridFrameEncoder        │  │
│  │ - MultiModalFusionLayer     │  │ - CNNAudioEncoder           │  │   - CNNBackbone             │  │
│  │ - VisionEncoder (ViT+CNN)   │  │ - RNNAudioEncoder           │  │   - VisionTransformerBranch │  │
│  │ - AudioEncoder              │  │ - TransformerEncoder        │  │ - TemporalEncoder           │  │
│  │ - Adaptive gating           │  │ - SpeechRecognitionModule   │  │ - ObjectTrackingModule      │  │
│  │                             │  │ - MusicAnalysisModule       │  │ - ActionRecognitionModule   │  │
│  │                             │  │ - AudioEmotionModule        │  │ - SceneUnderstandingModule  │  │
│  │                             │  │                             │  │ - MotionAnalysisModule      │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        REASONING & LEARNING                                         │
│                                                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  ┌─────────────────────────────┐ │
│  │                  ReasoningEngine (core/reasoning.py)          │  │   AGI Capabilities          │ │
│  │                                                               │  │  (modules/capabilities/)    │ │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐     │  │                             │ │
│  │  │    ReasoningStep    │  │  ChainOfThoughtReasoning    │     │  │ - IntegratedAGISystem       │ │
│  │  │                     │  │                             │     │  │   - AGI stages (0-6)        │ │
│  │  │ - Question encoder  │  │ - Multi-step reasoning      │     │  │   - Task routing            │ │
│  │  │ - Working memory    │  │ - Step selection            │     │  │   - Capability tracking     │ │
│  │  │ - Hypothesis gen    │  │ - Answer synthesis          │     │  │                             │ │
│  │  │ - Evidence integr   │  │                             │     │  │ - RealTimeLearningSystem    │ │
│  │  │ - Thought tracking  │  └─────────────────────────────┘     │  │   - FeedbackBuffer          │ │
│  │  └─────────────────────┘                                      │  │   - DynamicSkillAcquisition │ │
│  │                                                               │  │                             │ │
│  │  ┌─────────────────────────────────────────────────────┐      │  │ - ZeroShotConceptualSystem  │ │
│  │  │           MetaLearningController                    │      │  │   - ConceptualKnowledgeGraph│ │
│  │  │                                                     │      │  │   - Multi-hop reasoning     │ │
│  │  │ - Task encoding                                     │      │  │   - Analogy reasoning       │ │
│  │  │ - Few-shot adaptation                               │      │  │                             │ │
│  │  └─────────────────────────────────────────────────────┘      │  └─────────────────────────────┘ │
│  │                                                               │                                  │
│  │  ┌─────────────────────────────────────────────────────┐      │                                  │
│  │  │     RecursiveLanguageModel (RLM) (core/rlm/)        │      │                                  │
│  │  │                                                     │      │                                  │
│  │  │ - ContextStore: External context storage            │      │                                  │
│  │  │ - ContextTools: peek, grep, partition, summarize    │      │                                  │
│  │  │ - ToolSelector: Neural tool selection               │      │                                  │
│  │  │ - RecursiveCallManager: Spawn/aggregate subcalls    │      │                                  │
│  │  │ - Solves "context rot" for long documents           │      │                                  │
│  │  └─────────────────────────────────────────────────────┘      │                                  │
│  └───────────────────────────────────────────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      TOKEN SAMPLING                                                 │
│                                    (core/sampling.py)                                               │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                    TokenSampler                                             │    │
│  │                                                                                             │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐     │    │
│  │  │  Temperature    │  │    Top-K        │  │    Top-P        │  │ Repetition Penalty  │     │    │
│  │  │                 │  │                 │  │   (Nucleus)     │  │                     │     │    │
│  │  │ - Scale logits  │  │ - Keep top-k    │  │ - Cumulative    │  │ - Token history     │     │    │
│  │  │ - Control       │  │ - Filter rest   │  │   probability   │  │ - Penalty scoring   │     │    │
│  │  │   randomness    │  │ - Dynamic cutoff│  │ - Dynamic vocab │  │ - Configurable      │     │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────────┘     │    │
│  │                                                                                             │    │
│  │  ┌─────────────────┐  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │    Min-P        │  │                    Sampling Presets                             │   │    │
│  │  │                 │  │                                                                 │   │    │
│  │  │ - Relative      │  │  - Creative (temp=1.2, top_p=0.95, top_k=100)                   │   │    │
│  │  │   probability   │  │  - Precise (temp=0.3, top_p=0.9, top_k=20)                      │   │    │
│  │  │ - Quality       │  │  - Balanced (temp=0.7, top_p=0.9, top_k=50)                     │   │    │
│  │  │   threshold     │  │  - Deterministic (temp=0.0, greedy=true)                        │   │    │
│  │  └─────────────────┘  └─────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SUPPORT SYSTEMS                                                  │
│                                                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │     Ethics (core/ethics/)   │  │   Checkpoint Management     │  │   Data Pipeline (External)  │  │
│  │                             │  │ (core/checkpoint_manager.py)│  │                             │  │
│  │ - EthicalRewardModel        │  │                             │  │ See:                        │  │
│  │ - MultidimensionalBias      │  │ - SafeTensors format        │  │ github.com/AuralithAI/      │  │
│  │   Detector                  │  │ - Secure serialization      │  │   Auralith-Data-Pipeline    │  │
│  │   - Gender bias             │  │ - Version tracking          │  │                             │  │
│  │   - Racial bias             │  │ - Metadata storage          │  │ - Data collection           │  │
│  │   - Cultural bias           │  │ - Backward compatibility    │  │ - Tokenization              │  │
│  │ - FeedbackCollector         │  │                             │  │ - Preprocessing             │  │
│  │ - FairnessAnalyzer          │  │                             │  │ - Sharding                  │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘  │
│                                                                                                     │
│  ┌─────────────────────────────┐                                                                    │
│  │     Configuration           │                                                                    │
│  │     (config/)               │  Note: This repository is model-centric. All inputs are tensors.  │
│  │                             │  External integrations (web search, APIs) belong in separate      │
│  │ - AGIConfig                 │  application layers that interface with this model.               │
│  │   - Model architecture      │                                                                    │
│  │   - Training params         │                                                                    │
│  │   - Memory settings         │                                                                    │
│  │   - Quantum settings        │                                                                    │
│  │   - Ethics settings         │                                                                    │
│  │ - ImageConfig               │                                                                    │
│  │                             │                                                                    │
│  └─────────────────────────────┘                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          TRAINING                                                   │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                    train.py                                                 │    │
│  │                                                                                             │    │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────────┐  │    │
│  │  │      AGITrainer         │  │  Training Loop          │  │  Metrics Tracking           │  │    │
│  │  │                         │  │                         │  │  (core/evaluation.py)       │  │    │
│  │  │ - Model initialization  │  │ - Epoch-based training  │  │                             │  │    │
│  │  │ - Optimizer setup       │  │ - Batch processing      │  │ - Perplexity computation    │  │    │
│  │  │ - Parameter counting    │  │ - Gradient updates      │  │ - Token accuracy (top-1/5)  │  │    │
│  │  │ - Gradient clipping     │  │ - Checkpoint saving     │  │ - Gradient norm monitoring  │  │    │
│  │  └─────────────────────────┘  └─────────────────────────┘  │ - NaN/Inf/exploding detect  │  │    │
│  │                                                            │ - Structured JSON logging   │  │    │
│  │                                                            │ - Validation runner         │  │    │
│  │                                                            └─────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                  SCALABILITY OPTIONS                                        │    │
│  │                                                                                             │    │
│  │   Standard Mode (model_parallel=False)         Model Parallel Mode (model_parallel=True)   │    │
│  │   ─────────────────────────────────────        ────────────────────────────────────────    │    │
│  │                                                                                             │    │
│  │   create_rtdlm_agi(config)                     create_model_parallel_transformer()         │    │
│  │         │                                               │                                  │    │
│  │         ▼                                               ▼                                  │    │
│  │   ┌─────────────────────┐                     ┌─────────────────────┐                      │    │
│  │   │  Full AGI Model     │                     │ Sharded Transformer │                      │    │
│  │   │                     │                     │                     │                      │    │
│  │   │ - Consciousness     │                     │ - TensorParallel    │                      │    │
│  │   │ - Quantum           │                     │   Attention         │                      │    │
│  │   │ - Multimodal        │                     │ - TensorParallel    │                      │    │
│  │   │ - Reasoning         │                     │   MLP               │                      │    │
│  │   │ - Ethics            │                     │ - DeviceMesh        │                      │    │
│  │   │ - Memory Bank       │                     │   sharding          │                      │    │
│  │   └─────────────────────┘                     └─────────────────────┘                      │    │
│  │                                                                                             │    │
│  │   Best for: Single GPU/TPU                    Best for: Multi-GPU clusters                 │    │
│  │   Use when: Full AGI features needed          Use when: Model too large for one device     │    │
│  │                                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                            tests/test_runner.py                                             │    │
│  │                                                                                             │    │
│  │  - Test orchestration       - Benchmark mode          - Demo execution                      │    │
│  │  - 328 tests (20+ suites)   - Timeout handling        - Requirements check                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
                                           INPUT
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
             ┌──────────┐             ┌──────────┐             ┌──────────┐
             │   Text   │             │  Image   │             │  Audio   │
             └────┬─────┘             └────┬─────┘             └────┬─────┘
                  │                        │                        │
                  ▼                        ▼                        ▼
        ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
        │  Pre-tokenized  │      │  VisionEncoder  │      │  HybridAudio    │
        │  Input Tensor   │      │  (ViT + CNN)    │      │  Encoder        │
        └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
                 │                        │                        │
                 └────────────────────────┼────────────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │ MultiModalFusionLayer │
                              │ - CrossModalAttention │
                              │ - Adaptive Gating     │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │       TMSModel        │
                              │ - SelfAttention       │
                              │ - Transformer         │
                              │ - SparseMoE           │
                              │ - MemoryBank          │
                              └───────────┬───────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│   HybridArchitecture  │    │     ReasoningEngine   │    │    QuantumLayer       │
│                       │    │                       │    │                       │
│ - Traditional ML      │    │ - ChainOfThought      │    │ - QuantumSimulator    │
│ - Deep Learning       │    │ - MetaLearning        │    │ - VQC                 │
│ - Symbolic AI         │    │ - Working Memory      │    │ - QuantumAttention    │
│ - Probabilistic       │    │                       │    │                       │
└───────────┬───────────┘    └───────────┬───────────┘    └───────────┬───────────┘
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  Cognitive Core       │
                              │                       │
                              │ - Consciousness       │
                              │ - Scientific          │
                              │ - Creative            │
                              │ - Social-Emotional    │
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
           ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
           │ Ethics Module │     │ Web Integration│    │ Multi-Agent   │
           │               │     │               │     │ Consensus     │
           │ - Bias detect │     │ - DuckDuckGo  │     │ - Specialist  │
           │ - Fairness    │     │ - Wikipedia   │     │   agents      │
           │ - Reward      │     │ - Knowledge   │     │ - Voting      │
           └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
                   │                     │                     │
                   └─────────────────────┼─────────────────────┘
                                         │
                                         ▼
                              ┌───────────────────────┐
                              │     TokenSampler      │
                              │                       │
                              │ - Top-K / Top-P       │
                              │ - Temperature         │
                              │ - Repetition penalty  │
                              │ - Probability logging │
                              └───────────┬───────────┘
                                          │
                                          ▼
                                      OUTPUT
```

## Module Dependencies

```
rtdlm.py
├── core/model/model_tms.py
│   ├── core/model/model_module_self_attention.py  # Unified attention (RoPE, GQA, etc.)
│   ├── core/model/advanced_attention.py           # Attention implementations
│   ├── core/model/model_transformer_module.py
│   ├── core/model/sparse_moe.py
│   ├── core/model/memory_bank.py
│   └── core/ethics/reward_model.py
├── core/components/graph_neurons.py               # Graph neural components
├── modules/multimodal/fusion_module.py
├── modules/multimodal/hybrid_audio_module.py
├── modules/multimodal/hybrid_video_module.py
├── core/reasoning.py
├── core/sampling.py                               # Token sampling (dev utility)
├── core/evaluation.py                             # Evaluation metrics & logging
├── core/quantum/quantum_agi_core.py
├── core/quantum/quantum_readiness.py
├── config/agi_config.py
└── modules/hybrid_architecture/hybrid_integrator.py

train.py
├── Model components (all dependencies above)
├── config/agi_config.py
├── config/model_parallel_config.py      # Model parallelism settings
├── core/model_parallel.py               # Tensor/pipeline parallelism
├── core/training_utils.py               # Mixed precision, checkpointing
├── core/evaluation.py                   # Perplexity, gradient monitoring
├── core/benchmark_evaluation.py         # Production metrics (ECE, FLOPs, benchmarks)
├── core/gradient_accumulation.py        # Gradient accumulation for large batches
├── core/memory_profiler.py              # GPU memory profiling & optimization
└── core/checkpoint_manager.py

modules/capabilities/integrated_agi_system.py
├── modules/capabilities/real_time_learning.py
├── modules/capabilities/zero_shot_reasoning.py
└── core/quantum/quantum_readiness.py
```

## Evaluation System

RT-DLM provides a comprehensive evaluation system for training completeness.

### Metrics Tracked

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION SYSTEM                           │
│                    (core/evaluation.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │  EvaluationMetrics   │  │     GradientMonitor             │  │
│  │                      │  │                                 │  │
│  │  • Perplexity        │  │  • Global gradient norm         │  │
│  │  • Token accuracy    │  │  • Per-layer norms              │  │
│  │  • Top-5 accuracy    │  │  • NaN/Inf detection            │  │
│  │  • Entropy           │  │  • Exploding/vanishing detect   │  │
│  │  • Masking support   │  │  • Trend analysis               │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │  MetricLogger        │  │     ValidationRunner            │  │
│  │                      │  │                                 │  │
│  │  • JSON-lines format │  │  • Batched validation           │  │
│  │  • Console logging   │  │  • Metric aggregation           │  │
│  │  • Config storage    │  │  • Progress reporting           │  │
│  │  • Best metric track │  │  • Standard deviation           │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                TrainingEvaluator (High-Level API)        │   │
│  │                                                          │   │
│  │  Combines all components for easy training integration   │   │
│  │  • on_train_step() - Log metrics for each training step  │   │
│  │  • run_validation() - Periodic validation with logging   │   │
│  │  • summary() - Training summary with best metrics        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

See `core/evaluation.py` for the `TrainingEvaluator` API. Metrics tracked include perplexity, token accuracy, gradient norms, and throughput.

### Production Metrics (core/benchmark_evaluation.py)

Advanced metrics for production model evaluation:

```
┌─────────────────────────────────────────────────────────────────┐
│                   BENCHMARK EVALUATION                          │
│                (core/benchmark_evaluation.py)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │  PerplexityTracker   │  │     CalibrationTracker          │  │
│  │                      │  │                                 │  │
│  │  • Window-based PPL  │  │  • ECE (Expected Calibration)   │  │
│  │  • Rolling averages  │  │  • MCE (Maximum Calibration)    │  │
│  │  • Step tracking     │  │  • Bin-wise analysis            │  │
│  │  • Best PPL memory   │  │  • Reliability diagrams         │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │ ComputeEfficiency    │  │     BenchmarkEvaluator          │  │
│  │     Tracker          │  │                                 │  │
│  │                      │  │  • MMLU-style evaluation        │  │
│  │  • Tokens/second     │  │  • Multiple choice scoring      │  │
│  │  • FLOPs estimation  │  │  • Per-category breakdown       │  │
│  │  • Throughput stats  │  │  • Zero-shot/few-shot modes     │  │
│  │  • GPU utilization   │  │  • Chain-of-thought support     │  │
│  └──────────────────────┘  └─────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ProductionMetrics (Dataclass)               │   │
│  │                                                          │   │
│  │  Aggregates all metrics into a single container:         │   │
│  │  • perplexity, tokens_per_second, estimated_flops        │   │
│  │  • ece, mce, benchmark_accuracy, fairness_score          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**

| Tracker | Metric | Purpose |
|---------|--------|---------|
| `PerplexityTracker` | Perplexity | Measure model uncertainty |
| `CalibrationTracker` | ECE, MCE | Confidence reliability |
| `ComputeEfficiencyTracker` | Tokens/sec, FLOPs | Performance monitoring |
| `BenchmarkEvaluator` | Accuracy | Standard benchmark evaluation |

## Advanced Attention System

RT-DLM uses a unified attention system that supports multiple state-of-the-art attention mechanisms.

### Attention Types

| Type | Complexity | KV Cache | Best For |
|------|------------|----------|----------|
| **Standard MHA** | O(n²) | Full | General purpose, short sequences |
| **GQA** | O(n²) | 2-4x smaller | Fast inference, quality maintained |
| **MQA** | O(n²) | N/heads smaller | Maximum inference speed |
| **Sliding Window** | O(n × w) | Window only | Very long sequences (>8k tokens) |
| **Linear** | O(n) | None | Extreme length, approximate |

### Position Encoding

| Type | Description | Extrapolation |
|------|-------------|---------------|
| **RoPE** | Rotary Position Embedding | Excellent (recommended) |
| **Learned** | Traditional learned embeddings | Limited |
| **None** | No positional encoding | N/A |

Configure via `TMSModel(attention_type="gqa", num_kv_heads=2, position_encoding="rope", ...)`.

### Attention Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SelfAttentionModel (Unified)                        │
│                    (core/model/model_module_self_attention.py)              │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     advanced_attention.py                             │  │
│  │                                                                       │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │   │    RoPE     │  │     GQA     │  │  Sliding    │  │   Linear    │  │  │
│  │   │             │  │             │  │   Window    │  │  Attention  │  │  │
│  │   │ - Rotary    │  │ - Grouped   │  │             │  │             │  │  │
│  │   │   embed     │  │   KV heads  │  │ - O(n×w)    │  │ - O(n)      │  │  │
│  │   │ - Better    │  │ - 2-4x      │  │ - Local     │  │ - Kernel    │  │  │
│  │   │   extrap    │  │   smaller   │  │   context   │  │   approx    │  │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │      Spiking Attention          │  │        Pruning Manager          │   │
│  │                                 │  │                                 │   │
│  │ - Sparse activation             │  │ - Head usage tracking           │   │
│  │ - Threshold-based gating        │  │ - Dynamic pruning               │   │
│  │ - Energy efficient              │  │ - Compression analysis          │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## AGI-Scale Attention System

RT-DLM provides advanced attention mechanisms for AGI-level capabilities with long-context reasoning and deep memory interaction.

### AGI Attention Components

| Component | Purpose | Key Feature |
|-----------|---------|-------------|
| **Ring Attention** | Infinite context | Distributed attention across devices |
| **Cross-Memory Attention** | Memory interaction | LTM/STM/MTM interact via attention |
| **Hierarchical Memory Fusion** | Memory consolidation | Multi-level integration |
| **Infinite Context Attention** | Very long sequences | Hierarchical compression |

### Ring Attention

Ring Attention enables processing of arbitrarily long sequences by distributing attention across devices in a ring topology. Each device processes a local block while KV pairs are passed around the ring, achieving O(n/d × n) complexity where d is the number of devices.

Configuration: `config/agi_attention_config.py` - `AGIAttentionConfig.for_distributed()`

### Cross-Memory Attention

Instead of simple weighted sums, memory banks interact via cross-attention:

- **LTM ← STM**: Long-term memory queries short-term for recent updates
- **STM ← LTM**: Short-term queries long-term for persistent context
- **MTM ← LTM/STM**: Meta-task memory mediates between both for task fusion

Configuration: `config/agi_attention_config.py` - `MemoryFusionStrategy.CROSS_ATTENTION`

### Hierarchical Memory Fusion

Multi-level attention-based memory integration:

1. **Level 1**: Local self-attention within each memory bank
2. **Level 2**: Cross-attention between memory banks
3. **Level 3**: Global integration via importance-weighted fusion

Configuration: `config/agi_attention_config.py` - `MemoryFusionStrategy.HIERARCHICAL`

### Infinite Context Attention

Hierarchical compression for processing very long sequences:

- Processes input in chunks with local attention
- Compresses each chunk into summary tokens
- Maintains global context buffer of compressed summaries
- Complexity: O(chunk_size² + global_size²) instead of O(n²)

Configuration: `config/agi_attention_config.py` - `AGIAttentionConfig.for_long_context()`

### Module Structure

| File | Components |
|------|------------|
| `config/agi_attention_config.py` | AGIAttentionConfig, MemoryFusionStrategy, ContextStrategy |
| `core/model/agi_attention.py` | RingAttentionBlock, CrossMemoryAttention, HierarchicalMemoryFusion, InfiniteContextAttention, AGIAttention |

### Performance Comparison

| Feature | Legacy Mode | AGI Mode |
|---------|-------------|----------|
| Memory Interaction | Weighted sum | Cross-attention |
| Max Context | Fixed (sliding window) | Unlimited (ring) |
| Memory Relevance | Static weights | Dynamic attention |
| Consolidation | None | Hierarchical |

## Graph Neural Components

Graph-based neural components for relational reasoning.

### Components

| Component | Description |
|-----------|-------------|
| **GraphNeuron** | Basic graph neural unit with message passing |
| **GraphAttentionUnit** | Graph attention mechanism |
| **DynamicGraphBuilder** | Learns graph structure from data |
| **MultiHopGraphReasoner** | Multi-step relational reasoning |
| **GraphMoE** | Graph-structured mixture of experts |
| **GraphIntegratedTransformerBlock** | Combines transformer + graph attention |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Graph Neural Components                        │
│                  (core/components/graph_neurons.py)              │
│                                                                  │
│   Input Sequence                                                 │
│         │                                                        │
│         ▼                                                        │
│   ┌───────────────────┐                                          │
│   │ DynamicGraphBuilder│  ←── Learns adjacency from features     │
│   └─────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│   ┌───────────────────┐                                          │
│   │    GraphNeuron    │  ←── Message passing + aggregation       │
│   └─────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│   ┌───────────────────┐                                          │
│   │MultiHopGraphReasoner│ ←── K hops of graph reasoning          │
│   └─────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│   ┌───────────────────┐                                          │
│   │     GraphMoE      │  ←── Expert routing via graph structure  │
│   └─────────┬─────────┘                                          │
│             │                                                    │
│             ▼                                                    │
│      Output with                                                 │
│   relational context                                             │
└─────────────────────────────────────────────────────────────────┘
```

## Model Scale Presets

Pre-configured model sizes from tiny to production scale. Use `AGIConfig.from_preset("large")`.

| Preset | d_model | Heads | Layers | MoE Experts | ~Parameters |
|--------|---------|-------|--------|-------------|-------------|
| `tiny` | 256 | 4 | 4 | 4 | ~10M |
| `small` | 512 | 8 | 6 | 8 | ~50M |
| `base` | 768 | 12 | 12 | 8 | ~125M |
| `large` | 1024 | 16 | 24 | 16 | ~350M |
| `xlarge` | 2048 | 32 | 32 | 32 | ~1.3B |
| `xxlarge` | 4096 | 64 | 48 | 64 | ~7B |

### Memory Requirements per Preset

Estimated GPU memory for training with different configurations:

| Preset | Params | FP32 Training | FP16 Training | FP16 + Checkpointing | Recommended GPU |
|--------|--------|---------------|---------------|----------------------|-----------------|
| `tiny` | 10M | 0.2 GB | 0.1 GB | 0.1 GB | Any GPU (4GB+) |
| `small` | 50M | 0.8 GB | 0.5 GB | 0.4 GB | RTX 3060 (8GB) |
| `base` | 125M | 2.0 GB | 1.2 GB | 1.0 GB | RTX 3070 (8GB) |
| `large` | 350M | 5.5 GB | 3.3 GB | 2.8 GB | RTX 3090 (16GB) |
| `xlarge` | 1.3B | 20.3 GB | 12.2 GB | 10.2 GB | RTX 4090 (24GB) or A100 (40GB) |
| `xxlarge` | 7B | 109.4 GB | 65.6 GB | 54.7 GB | A100 (80GB) or Multi-GPU |

**Notes:**
- FP32 Training: Full precision, includes params + optimizer + gradients + activations
- FP16 Training: Mixed precision with FP32 optimizer states
- FP16 + Checkpointing: Gradient checkpointing reduces activation memory by ~65%
- Actual memory varies with batch size and sequence length
- For larger models, use gradient accumulation or model parallelism

### Gradient Accumulation

For limited GPU memory, use gradient accumulation to achieve larger effective batch sizes:

| Target Batch Size | Micro Batch | Accumulation Steps | GPU Memory |
|-------------------|-------------|-------------------|------------|
| 64 | 8 | 8 | Low |
| 128 | 16 | 8 | Medium |
| 256 | 8 | 32 | Low |
| 512 | 32 | 16 | High |

Use `core.gradient_accumulation.recommend_accumulation_steps()` to find optimal settings.

## Scalability & Training Modes

RT-DLM supports multiple training modes depending on your hardware and model size requirements.

### Training Mode Selection

```
                              AGIConfig Settings
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
           model_parallel=False   distributed=True  model_parallel=True
                    │                │                │
                    ▼                ▼                ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
            │   STANDARD  │   │    DATA     │   │   MODEL     │
            │   TRAINING  │   │  PARALLEL   │   │  PARALLEL   │
            └─────────────┘   └─────────────┘   └─────────────┘
                    │                │                │
                    ▼                ▼                ▼
        create_rtdlm_agi()    pmap across       create_model_parallel_
                │             devices           transformer()
                ▼                │                │
        Full AGI Model           ▼                ▼
        - Consciousness    Same model       Sharded Model
        - Quantum          replicated       - TensorParallelLinear
        - Multimodal       per device       - TensorParallelAttention
        - Reasoning                         - TensorParallelMLP
        - Ethics                            - DeviceMesh
```

### Training Mode Summary

| Mode | Use Case | Config |
|------|----------|--------|
| **Standard** | Single GPU, model fits in memory | Default |
| **Data Parallel** | Multiple GPUs, faster training | `distributed_training=True` |
| **Model Parallel** | Model too large for single device | `model_parallel=True` |

The full AGI model includes ConsciousnessSimulator, QuantumAGICore, MultiModalRTDLM, ReasoningEngine, EthicalRewardModel, and TMSModel with MemoryBank.

**Note**: Model parallel mode uses a simplified transformer architecture optimized for sharding.

### Model Parallel Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ModelParallelTransformer                     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      DeviceMesh                           │  │
│  │   Device 0    Device 1    Device 2    Device 3            │  │
│  │      │           │           │           │                │  │
│  │      └───────────┴───────────┴───────────┘                │  │
│  │                        │                                  │  │
│  │            Tensor Parallel Axis ("tensor")                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  For each transformer layer:                                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ TensorParallelAttention                                 │    │
│  │                                                         │    │
│  │  Q, K, V projections: Column-parallel (split output)    │    │
│  │  Each device computes local_num_heads = num_heads/N     │    │
│  │  Output projection: Row-parallel (all-reduce sum)       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ TensorParallelMLP                                       │    │
│  │                                                         │    │
│  │  FC1: Column-parallel (split d_ff across devices)       │    │
│  │  FC2: Row-parallel (all-reduce to combine)              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Quantum Simulation Scalability

> **⚠️ IMPORTANT: Classical Simulation Only**
>
> The quantum modules in RT-DLM are **classical simulations** of quantum algorithms, NOT actual 
> quantum hardware execution. These modules use NumPy/JAX to mathematically simulate quantum 
> gates (CNOT, Hadamard, Phase, rotation gates) on classical CPUs/GPUs.
>
> **Current Status:**
> - Research exploration of quantum-inspired attention and optimization
> - Classical approximation of quantum concepts (superposition, entanglement)
> - No quantum hardware integration
>
> **Future Quantum Hardware Integration:**
> - IBM Qiskit, AWS Braket, or Google Cirq for real quantum processors
> - Timeline: Real quantum utility for AI is likely 5-10+ years away
> - Architecture designed to swap simulation for hardware when available
>
> **Disabling Quantum Simulation:**
> Set `quantum_layers=0` in AGIConfig for faster training without quantum overhead.

For quantum simulation, tensor networks enable 100+ qubit simulation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quantum Simulation Modes                     │
│                                                                 │
│  Standard (quantum_max_qubits ≤ 30)    Extended (100+ qubits)   │
│  ─────────────────────────────────     ─────────────────────    │
│                                                                 │
│  Full state vector                     Tensor Network Approx    │
│  Memory: O(2^n)                        Memory: O(n × χ²)        │
│                                                                 │
│  QuantumSimulator                      TensorNetworkQuantum     │
│       │                                Simulator                │
│       ▼                                     │                   │
│  quantum_readiness.py                       ▼                   │
│                                        tensor_network.py        │
│                                        - MatrixProductState     │
│                                        - TreeTensorNetwork      │
└─────────────────────────────────────────────────────────────────┘
```

#### Quantum Simulation Cost Analysis

| Qubits | Mode | Memory | Parameters | Recommendation |
|--------|------|--------|------------|----------------|
| ≤16 | Full State | 1 MB | ~200 | Development/testing |
| 17-24 | Full State | 256 MB - 16 GB | ~300 | Research experiments |
| 25-30 | Full State + Sparse | 16 GB+ | ~400 | Requires high-memory GPU |
| 30+ | Tensor Network | O(n×χ²) ≈ 6.4 MB | ~500 | Required for 30+ qubits |

Use `estimate_quantum_overhead()` from `core.quantum` to get memory/compute estimates programmatically. Set `quantum_layers=0` to disable quantum simulation for faster training.

### Configuration Summary

The system uses a **unified scalable training approach**  - **ONE model** that supports both data and model parallelism through device mesh configuration:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Scalable Training                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ScalableMesh                         │    │
│  │                                                         │    │
│  │  Configuration:                                         │    │
│  │  - data_parallel_size: N devices for batch splitting    │    │
│  │  - tensor_parallel_size: M devices for weight sharding  │    │
│  │  - pipeline_parallel_size: P stages for layer splitting │    │
│  │                                                         │    │
│  │  Automatically handles:                                 │    │
│  │  - Parameter sharding specs                             │    │
│  │  - Gradient synchronization                             │    │
│  │  - Memory estimation and recommendations                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  RTDLMAGISystem                         │    │
│  │              (Single Unified Model)                     │    │
│  │                                                         │    │
│  │  Same model used for ALL configurations:                │    │
│  │  - Single device training                               │    │
│  │  - Multi-device data parallelism                        │    │
│  │  - Multi-device tensor parallelism                      │    │
│  │  - Combined data + tensor parallelism                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

| Config Flag | Effect | Implementation |
|-------------|--------|----------------|
| `model_parallel=False` | Standard single-device training | `create_rtdlm_agi()` |
| `model_parallel=True` | Tensor parallelism via ScalableMesh | `create_rtdlm_agi()` + mesh sharding |
| `distributed_training=True` | Data parallelism via ScalableMesh | `create_rtdlm_agi()` + replicated params |
| Both `True` | Combined parallelism (production-scale) | `create_rtdlm_agi()` + 2D mesh |
| `mixed_precision=True` | FP16/BF16 compute | Applied to unified model |
| `gradient_checkpointing=True` | Memory efficiency | Applied to unified model |

**Key Design Principle**: Unlike having separate models for different parallelism modes, we use ONE unified `RTDLMAGISystem` model. The `ScalableMesh` class handles how parameters are distributed across devices, making the same model work at any scale.

### Distributed Training Utilities

The `core.scalable_training` module provides production utilities:

| Function | Purpose |
|----------|---------|
| `estimate_model_memory(params)` | Estimate GPU memory requirements |
| `recommend_parallelism(memory_gb, device_gb, num_devices)` | Get optimal parallelism strategy |
| `profile_collective_communication(mesh)` | Measure all-reduce latency/bandwidth |
| `validate_distributed_setup(mesh)` | Verify multi-device configuration |
| `unreplicate_params(params)` | Extract single copy from replicated state |

## Graph-Based Neural Components

RT-DLM includes graph neural network components for enhanced relational reasoning:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Graph Neural Architecture                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 DynamicGraphBuilder                      │    │
│  │                                                         │    │
│  │  Builds graphs dynamically from sequence embeddings:    │    │
│  │  - Edge prediction via learned bilinear scoring         │    │
│  │  - Automatic self-loop addition                         │    │
│  │  - Configurable edge threshold                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    GraphNeuron                           │    │
│  │                                                         │    │
│  │  Graph attention with residual connections:             │    │
│  │  - Multi-head graph attention                           │    │
│  │  - Layer normalization                                  │    │
│  │  - Optional FFN layer                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MultiHopGraphReasoner                       │    │
│  │                                                         │    │
│  │  Multi-hop reasoning for chain-of-thought:              │    │
│  │  - Configurable number of hops                          │    │
│  │  - Query-guided attention paths                         │    │
│  │  - Tracks reasoning trajectories                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   GraphMoE                               │    │
│  │                                                         │    │
│  │  Mixture-of-Experts with relational routing:            │    │
│  │  - Graph-based expert selection                         │    │
│  │  - Neighbor-aware routing decisions                     │    │
│  │  - Combined local and relational scores                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Graph Configuration

Configure graph neurons via `GraphConfig` in `core.components`. Key parameters: `d_model`, `num_heads`, `max_nodes`, `edge_threshold`, and `num_hops`.

## Advanced MoE Features

The SparseMoE module includes advanced features for better expert utilization:

- **Router Jitter**: Multiplicative noise during training to prevent expert collapse
- **Capacity Factor Loss**: Prevents expert overflow by penalizing unbalanced loads
- **Adaptive Gating**: Context-aware routing with expert affinity prediction
- **Dynamic Load Balancing**: Historical usage tracking for balanced expert selection

## Speculative Decoding

For faster inference, the `core.sampling` module provides `SpeculativeDecoder` and `SelfSpeculativeDecoder` with draft-verify pipeline for 2-3x generation speedup.

## Directory Structure

```
RT-DLM/
├── config/                          # Configuration
│   ├── __init__.py
│   ├── agi_config.py                # AGI system configuration
│   ├── model_parallel_config.py     # Model parallelism settings
│   ├── tensor_network_config.py     # Tensor network settings
│   └── image_config.py              # Image generation config
│
├── core/                            # Core components
│   ├── __init__.py
│   ├── sampling.py                  # Token sampling (Top-K, Top-P, etc.)
│   ├── reasoning.py                 # Reasoning engine
│   ├── checkpoint_manager.py        # SafeTensors checkpoint management
│   ├── scalable_training.py         # Production-ready unified parallelism (ScalableMesh)
│   ├── model_parallel.py            # Legacy tensor/pipeline parallelism layers
│   ├── training_utils.py            # Mixed precision, gradient checkpointing
│   │
│   ├── model/                       # Neural architecture
│   │   ├── model_tms.py             # TMS model
│   │   ├── model_module_self_attention.py
│   │   ├── model_transformer_module.py
│   │   ├── sparse_moe.py            # Mixture of Experts
│   │   ├── memory_bank.py           # Three-tier memory
│   │   └── secure_tms.py            # Security wrapper
│   │
│   ├── quantum/                     # Quantum-inspired modules
│   │   ├── quantum_agi_core.py      # Quantum attention, memory
│   │   ├── quantum_readiness.py     # VQC, QuantumSimulator
│   │   ├── extended_quantum_sim.py  # 64+ qubit simulation
│   │   └── tensor_network.py        # MPS/TTN for 100+ qubits
│   │
│   ├── ethics/                      # Ethical AI
│   │   ├── feedback_collector.py
│   │   ├── ethical_adaptation.py
│   │   └── reward_model.py
│   │
│   ├── agi/                         # AGI sub-components
│   │
│   └── components/                  # Reusable components
│       ├── __init__.py
│       ├── reusable_components.py   # Attention, FFN, transformers
│       └── graph_neurons.py         # Graph neural components
│
├── modules/                         # Feature modules
│   ├── __init__.py
│   │
│   ├── multimodal/                  # Multimodal processing
│   │   ├── fusion_module.py
│   │   ├── hybrid_audio_module.py
│   │   └── hybrid_video_module.py
│   │
│   ├── hybrid_architecture/         # Multi-paradigm learning
│   │   └── hybrid_integrator.py
│   │
│   └── capabilities/                # AGI capabilities
│       ├── integrated_agi_system.py
│       ├── real_time_learning.py
│       ├── advanced_algorithms.py   # EWC, continual learning
│       └── zero_shot_reasoning.py
│
├── tests/                           # Test suite (328 tests)
│   ├── test_framework.py            # Main test framework
│   ├── test_runner.py               # Test orchestration
│   ├── test_model_parallel.py       # Model parallelism tests
│   ├── test_tensor_network.py       # Tensor network tests
│   ├── test_config.py
│   ├── system_validator.py
│   ├── test_model/                  # Model tests
│   └── demo/                        # Demo scripts
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # This file
│   ├── SAMPLING.md                  # Sampling strategies
│   └── QUICKSTART.md                # Getting started
│
├── rtdlm.py                         # Main AGI model
├── train.py                         # Training script
├── inference.py                     # Inference script
├── install_dependencies.py          # Dependency installer
├── requirements.txt                 # Python dependencies
├── LICENSE                          # License file
└── README.md                        # Project readme
```

> **Note**: Tokenization and data processing are handled by [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).

## Technology Stack

```
┌────────────────────────────────────────────────────────────────┐
│                        FRAMEWORK LAYER                         │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ JAX 0.4.35  │  │Haiku 0.0.13 │  │   Optax     │             │
│  │             │  │             │  │             │             │
│  │ - JIT       │  │ - hk.Module │  │ - AdamW     │             │
│  │ - vmap      │  │ - hk.Embed  │  │ - Cosine    │             │
│  │ - grad      │  │ - hk.Linear │  │   schedule  │             │
│  │ - pmap      │  │ - hk.Conv2D │  │ - Gradient  │             │
│  │             │  │ - hk.GRU    │  │   clipping  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   FAISS     │  │   NumPy     │  │ SafeTensors │             │
│  │             │  │             │  │             │             │
│  │ - IndexFlat │  │ - Arrays    │  │ - Secure    │             │
│  │ - Retrieval │  │ - Math ops  │  │   weights   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │  librosa    │  │  OpenCV     │                              │
│  │             │  │             │                              │
│  │ - Audio     │  │ - Video     │                              │
│  │ - MFCC      │  │ - Frames    │                              │
│  └─────────────┘  └─────────────┘                              │
└────────────────────────────────────────────────────────────────┘
```

## File Statistics

| Category | Location | Files | Status |
|----------|----------|-------|--------|
| Model Core | `core/` | 6+ | Complete |
| Quantum | `quantum/` | 2 | Complete |
| Ethics | `ethics/` | 3 | Complete |
| TMS Block | `TMS_block/` | 6 | Complete |
| MoE Block | `moe_block/` | 2 | Complete |
| Transformer | `transformer_block/` | 4 | Complete |
| Hybrid Architecture | `hybrid_architecture/` | 1 | Complete |
| Multimodal | `multimodal/` | 3 | Complete |
| Reasoning | `reasoning/` | 1 | Complete |
| RLM | `core/rlm/` | 5 | Complete |
| Configuration | `config/` | 4 | Complete |
| Training | `train.py` | 1 | Complete |
| Tests | `tests/` | tests | Complete |

> **Note**: Tokenization and data processing have been moved to [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).

## Recursive Language Model (RLM)

The RLM module (`core/rlm/`) implements the Recursive Language Model architecture inspired by MIT research ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)). It solves the "context rot" problem where LLMs degrade when processing long documents even within context window limits.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RecursiveLanguageModel (RLM)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │   ContextStore      │    │           ContextTools                    │   │
│  │   (External REPL)   │    │                                           │   │
│  │                     │    │  ┌─────────┐ ┌─────────┐ ┌───────────┐   │   │
│  │  - Variables Dict   │◄───│  │  peek() │ │  grep() │ │partition()│   │   │
│  │  - Context Registry │    │  └─────────┘ └─────────┘ └───────────┘   │   │
│  │  - Chunk Manager    │    │  ┌─────────┐ ┌─────────┐ ┌───────────┐   │   │
│  └─────────────────────┘    │  │summarize│ │ count() │ │  filter() │   │   │
│            ▲                │  └─────────┘ └─────────┘ └───────────┘   │   │
│            │                └──────────────────────────────────────────┘   │
│            │                                     │                         │
│            │                                     ▼                         │
│  ┌─────────┴─────────────────────────────────────────────────────────────┐ │
│  │                    ToolSelector (Neural Network)                       │ │
│  │  - Encodes query + context metadata + recursion state                  │ │
│  │  - Outputs tool probabilities + termination probability                │ │
│  │  - Extracts tool parameters from learned heads                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      RecursiveCallManager                               │ │
│  │  - Tracks recursion depth and tool budget                               │ │
│  │  - Spawns parallel subcalls for chunk processing                        │ │
│  │  - Aggregates results (weighted mean, concat, simple)                   │ │
│  │  - Caches results for repeated queries                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| External Context | Context stored as variables, not in model input |
| Tool-Based Exploration | peek, grep, partition, summarize, count, filter |
| Recursive Decomposition | Spawn subcalls on partitioned chunks |
| Neural Tool Selection | Learned policy for tool selection |
| Parallel Subcalls | Process multiple chunks concurrently |
| Direct Pass Fallback | Skip RLM for short contexts |

RLM is configurable via `RLMConfig` and integrates with `ReasoningEngine` through `AGIConfig`.
