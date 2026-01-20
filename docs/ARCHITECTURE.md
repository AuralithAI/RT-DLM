# RT-DLM Architecture

This document describes the model architecture for training.

> **Note**: Data collection, tokenization, and processing are handled by [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).

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

### Usage in Training Loop

```python
from core.evaluation import TrainingEvaluator

# Initialize
evaluator = TrainingEvaluator(
    vocab_size=50257,
    log_dir="./logs",
    experiment_name="my-run",
    validate_every_n_steps=1000,
)

# Training loop
for step, batch in enumerate(dataloader):
    loss, logits, grads = train_step(params, batch)
    
    # Log all metrics
    evaluator.on_train_step(
        step=step,
        loss=loss,
        logits=logits,
        targets=batch['targets'],
        learning_rate=current_lr,
        grads=grads,  # For gradient health monitoring
    )
    
    # Validation
    if evaluator.should_validate(step):
        evaluator.run_validation(model_fn, params, val_data, step)

# Summary
print(evaluator.summary())
```

### Log Output Format

```
Step     100 | loss: 4.2341 | ppl: 68.92 | acc: 0.1234 | lr: 1.00e-04 | grad_norm: 0.8234 | 12543 tok/s
Step     200 | loss: 3.8921 | ppl: 48.98 | acc: 0.1567 | lr: 9.95e-05 | grad_norm: 0.7891 | 13102 tok/s
...
============================================================
VALIDATION at step 1000
  Loss:       3.4521 (±0.0234)
  Perplexity: 31.56 (±2.34)
  Accuracy:   0.2134
  Tokens:     512,000
  Time:       45.2s
============================================================
```

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

### Configuration Example

```python
from core.model import TMSModel

# Standard MHA with RoPE (default, backward compatible)
model = TMSModel(d_model=512, num_heads=8, attention_type="standard", ...)

# GQA for 4x faster inference with minimal quality loss
model = TMSModel(d_model=512, num_heads=8, attention_type="gqa", num_kv_heads=2, ...)

# Sliding window for very long documents
model = TMSModel(d_model=512, num_heads=8, attention_type="sliding", sliding_window_size=4096, ...)

# Linear attention for extreme efficiency
model = TMSModel(d_model=512, num_heads=8, attention_type="linear", ...)
```

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

Pre-configured model sizes from tiny to production scale.

| Preset | d_model | Heads | Layers | MoE Experts | ~Parameters |
|--------|---------|-------|--------|-------------|-------------|
| `tiny` | 256 | 4 | 4 | 4 | ~10M |
| `small` | 512 | 8 | 6 | 8 | ~50M |
| `base` | 768 | 12 | 12 | 8 | ~125M |
| `large` | 1024 | 16 | 24 | 16 | ~350M |
| `xlarge` | 2048 | 32 | 32 | 32 | ~1.3B |
| `xxlarge` | 4096 | 64 | 48 | 64 | ~7B |

```python
from config import AGIConfig

# Load preset
config = AGIConfig.from_preset("large")

# Customize from preset
config = AGIConfig.from_preset("base")
config.attention_type = "gqa"
config.num_kv_heads = 4
```

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

### 1. Standard Training (Default)

**Use when**: Single GPU/TPU, model fits in memory

```python
config = AGIConfig(
    model_parallel=False,      # Default
    distributed_training=False
)
trainer = AGITrainer(config)
# Uses: create_rtdlm_agi(config) → Full AGI model with all features
```

The full AGI model includes:
- ConsciousnessSimulator
- QuantumAGICore
- MultiModalRTDLM
- ReasoningEngine
- EthicalRewardModel
- TMSModel with MemoryBank

### 2. Data Parallel Training

**Use when**: Multiple GPUs, want faster training with same model

```python
config = AGIConfig(
    distributed_training=True,
    num_devices=4,
    data_parallel=True
)
trainer = AGITrainer(config)
# Uses: pmap to replicate model across devices
# Each device processes different data batches
```

### 3. Model Parallel Training

**Use when**: Model too large to fit on single device

```python
config = AGIConfig(
    model_parallel=True,
    num_devices=8
)
trainer = AGITrainer(config)
# Uses: create_model_parallel_transformer(config, device_mesh)
# Model layers are split across devices
```

**Note**: Model parallel mode uses a simplified transformer architecture (without consciousness, quantum, etc.) because sharding complex nested modules is challenging. This mode is for training very large base models that can later be extended.

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

```python
from core.components import GraphConfig, GraphNeuron, create_graph_neuron

# Configure graph neurons
config = GraphConfig(
    d_model=384,
    num_heads=8,
    max_nodes=64,
    edge_threshold=0.3,
    num_hops=3,
    enable_relational_routing=True,
)

# Create graph neuron
graph_neuron = create_graph_neuron(config)
```

## Model Scale Presets

RT-DLM provides pre-configured model scales from development to production:

| Preset | d_model | layers | heads | experts | ~Params | Use Case |
|--------|---------|--------|-------|---------|---------|----------|
| `tiny` | 256 | 6 | 4 | 4 | 50M | Testing |
| `small` | 384 | 12 | 8 | 8 | 150M | Development |
| `base` | 768 | 12 | 12 | 8 | 350M | Fine-tuning |
| `large` | 1024 | 24 | 16 | 16 | 1B | Production |
| `xlarge` | 2048 | 32 | 32 | 32 | 7B | Advanced |
| `xxlarge` | 4096 | 48 | 64 | 64 | 70B | SOTA |

```python
from config.agi_config import AGIConfig

# Create config from preset
config = AGIConfig.from_preset('large')

# List available presets
AGIConfig.list_presets()

# Customize preset
config = AGIConfig.from_preset('xlarge', learning_rate=5e-5, batch_size=64)
```

## Advanced MoE Features

The SparseMoE module includes advanced features for better expert utilization:

- **Router Jitter**: Multiplicative noise during training to prevent expert collapse
- **Capacity Factor Loss**: Prevents expert overflow by penalizing unbalanced loads
- **Adaptive Gating**: Context-aware routing with expert affinity prediction
- **Dynamic Load Balancing**: Historical usage tracking for balanced expert selection

## Speculative Decoding

For faster inference, the sampling module provides speculative decoding:

```python
from core.sampling import SpeculativeDecoder

# Setup speculative decoding with draft model
decoder = SpeculativeDecoder(
    target_forward_fn=large_model.apply,
    draft_forward_fn=small_model.apply,
    num_speculative_tokens=4,
)

# Generate with 2-3x speedup
tokens = decoder.generate(target_params, draft_params, initial_tokens, rng_key)
```

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
| Configuration | `config/` | 3 | Complete |
| Training | `train.py` | 1 | Complete |
| Tests | `tests/` | tests | Complete |

> **Note**: Tokenization and data processing have been moved to [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).
