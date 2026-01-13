# RT-DLM Architecture

This document describes the model architecture for training and inference.

> **Note**: Data collection and processing are handled by [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).

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
│  │ - TransformerModel      │    │ │  │ - TraditionalML     │    │ │  │ - CNOT gate         │    │
│  │ - SparseMoE             │    │ │  │   - SVMLikeClassifier│   │ │  │ - Hadamard gate     │    │
│  │ - EthicalRewardModel    │    │ │  │   - RandomForestLike│    │ │  │ - Rotation gates    │    │
│  └─────────────────────────┘    │ │  │   - NaiveBayesLike  │    │ │  │ - 32-qubit max      │    │
│                                 │ │  │                     │    │ │  └─────────────────────┘    │
│  ┌─────────────────────────┐    │ │  │ - DeepLearning      │    │ │                             │
│  │      MemoryBank         │    │ │  │   - CNNBranch       │    │ │  ┌─────────────────────┐    │
│  │                         │    │ │  │   - RNNBranch       │    │ │  │ VariationalQuantum  │    │
│  │ - LTM (Long-Term)       │    │ │  │   - TransformerBranch│   │ │  │     Circuit         │    │
│  │ - STM (Short-Term)      │    │ │  │                     │    │ │  │                     │    │
│  │ - MTM (Meta-Task)       │    │ │  │ - SymbolicReasoning │    │ │  │ - build_layers()    │    │
│  │ - FAISS indexing        │    │ │  │ - Probabilistic     │    │ │  │ - Parameterized     │    │
│  │ - Adaptive forgetting   │    │ │  └─────────────────────┘    │ │  │ - Trainable params  │    │
│  └─────────────────────────┘    │ │                             │ │  └─────────────────────┘    │
│                                 │ │  ┌─────────────────────┐    │ │                             │
│  ┌─────────────────────────┐    │ │  │   EnsembleFusion    │    │ │  ┌─────────────────────┐    │
│  │      SparseMoE          │    │ │  │                     │    │ │  │ QuantumInspired     │    │
│  │                         │    │ │  │ - Cross-paradigm    │    │ │  │    Attention        │    │
│  │ - AdaptiveGatingNetwork │    │ │  │   interaction       │    │ │  │                     │    │
│  │ - Top-K expert routing  │    │ │  │ - Outer-product     │    │ │  │ - Superposition     │    │
│  │ - Load balancing loss   │    │ │  │   attention         │    │ │  │ - Entanglement      │    │
│  │ - Specialization loss   │    │ │  │ - Adaptive fusion   │    │ │  │ - Quantum gates     │    │
│  │ - Dynamic capacity      │    │ │  └─────────────────────┘    │ │  └─────────────────────┘    │
│  └─────────────────────────┘    │ │                             │ │                             │
└─────────────────────────────────┘ │  ┌─────────────────────┐    │ │  ┌─────────────────────┐    │
                                    │  │  MultiAgentConsensus│    │ │  │QuantumNeuralNetwork │    │
                                    │  │                     │    │ │  │                     │    │
                                    │  │ - SpecialistAgents  │    │ │  │ - QuantumEncoder    │    │
                                    │  │   - Reasoning       │    │ │  │ - QuantumDecoder    │    │
                                    │  │   - Creativity      │    │ │  │ - QuantumAttention  │    │
                                    │  │   - Analysis        │    │ │  └─────────────────────┘    │
                                    │  │   - Synthesis       │    │ └─────────────────────────────┘
                                    │  │ - Weighted voting   │    │
                                    │  │ - Consensus loop    │    │
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
│  │     Ethics (core/ethics/)   │  │  External Integration       │  │    Tokenization             │  │
│  │                             │  │  (modules/integrations/)    │  │    (tokenization/)          │  │
│  │ - EthicalRewardModel        │  │                             │  │                             │  │
│  │ - MultidimensionalBias      │  │ - WebSearchModule           │  │ - MultiModalTokenizer       │  │
│  │   Detector                  │  │   - DuckDuckGo search       │  │ - TokenizationConfig        │  │
│  │   - Gender bias             │  │   - Wikipedia search        │  │ - ModalityType enum         │  │
│  │   - Racial bias             │  │   - Real embeddings         │  │ - SentencePiece integration │  │
│  │   - Cultural bias           │  │   - vocab_size: 50000       │  │ - Text/Image/Audio/Video    │  │
│  │ - FeedbackCollector         │  │                             │  │ - PDF/XML/JSON support      │  │
│  │ - FairnessAnalyzer          │  │ - HybridKnowledgeIntegration│  │                             │  │
│  └─────────────────────────────┘  │   - Knowledge fusion        │  └─────────────────────────────┘  │
│                                   │   - Relevance scoring       │                                   │
│                                   └─────────────────────────────┘                                   │
│                                                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐  │
│  │     Configuration           │  │   Checkpoint Management     │  │   Data Pipeline (External)  │  │
│  │     (config/)               │  │ (core/checkpoint_manager.py)│  │                             │  │
│  │                             │  │                             │  │ See:                        │  │
│  │ - AGIConfig                 │  │ - SafeTensors format        │  │ github.com/AuralithAI/      │  │
│  │   - Model architecture      │  │ - Secure serialization      │  │   Auralith-Data-Pipeline    │  │
│  │   - Training params         │  │ - Version tracking          │  │                             │  │
│  │   - Memory settings         │  │ - Metadata storage          │  │ - Data collection           │  │
│  │   - Quantum settings        │  │ - Backward compatibility    │  │ - Preprocessing             │  │
│  │   - Ethics settings         │  │                             │  │ - Sharding                  │  │
│  │ - ImageConfig               │  │                             │  │ - Storage backends          │  │
│  │                             │  │                             │  │                             │  │
│  └─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  TRAINING & INFERENCE                                               │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                    train.py                                                 │    │
│  │                                                                                             │    │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────────┐  │    │
│  │  │      AGITrainer         │  │  Training Loop          │  │  Metrics Tracking           │  │    │
│  │  │                         │  │                         │  │                             │  │    │
│  │  │ - Model initialization  │  │ - Epoch-based training  │  │ - Training losses           │  │    │
│  │  │ - Optimizer setup       │  │ - Batch processing      │  │ - Validation losses         │  │    │
│  │  │ - Parameter counting    │  │ - Gradient updates      │  │ - Reasoning accuracies      │  │    │
│  │  │                         │  │ - Checkpoint saving     │  │ - Consciousness coherence   │  │    │
│  │  └─────────────────────────┘  └─────────────────────────┘  │ - Multimodal alignment      │  │    │
│  │                                                            └─────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                                 inference.py                                                │    │
│  │                                                                                             │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                      RT_DLM_AGI_Assistant                                           │    │    │
│  │  │                                                                                     │    │    │
│  │  │ - preprocess_input()     - think_step_by_step()     - scientific_inquiry()          │    │    │
│  │  │ - generate_response()    - creative_generation()    - interactive_session()         │    │    │
│  │  │ - checkpoint loading     - conversation history     - knowledge base                │    │    │
│  │  │ - TokenSampler integration (Top-K, Top-P, temperature, repetition penalty)          │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                            tests/test_runner.py                                             │    │
│  │                                                                                             │    │
│  │  - Test orchestration       - Benchmark mode          - Demo execution                      │    │
│  │  - 244 tests (19 suites)    - Timeout handling        - Requirements check                  │    │
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
        │  MultiModal     │      │  VisionEncoder  │      │  HybridAudio    │
        │  Tokenizer      │      │  (ViT + CNN)    │      │  Encoder        │
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
│   ├── core/model/model_module_self_attention.py
│   ├── core/model/model_transformer_module.py
│   ├── core/model/sparse_moe.py
│   ├── core/model/memory_bank.py
│   └── core/ethics/reward_model.py
├── modules/multimodal/fusion_module.py
├── modules/multimodal/hybrid_audio_module.py
├── modules/multimodal/hybrid_video_module.py
├── core/reasoning.py
├── core/quantum/quantum_agi_core.py
├── core/quantum/quantum_readiness.py
├── config/agi_config.py
│   └── modules/tokenization/multimodal_tokenizer.py
├── modules/integrations/web_integration.py
└── modules/hybrid_architecture/hybrid_integrator.py

train.py
├── Model components (all dependencies above)
├── config/agi_config.py
├── config/train_config.py
└── data_processing/data_utils.py

inference.py
├── Model components (all dependencies above)
├── config/agi_config.py
├── data_processing/data_utils.py
└── core/inference_engine.py

modules/capabilities/integrated_agi_system.py
├── modules/capabilities/real_time_learning.py
├── modules/capabilities/zero_shot_reasoning.py
└── core/quantum/quantum_readiness.py
```

## Directory Structure

```
RT-DLM/
├── config/                          # Configuration
│   ├── __init__.py
│   ├── agi_config.py                # AGI system configuration
│   └── image_config.py              # Image generation config
│
├── core/                            # Core components
│   ├── __init__.py
│   ├── sampling.py                  # Token sampling (Top-K, Top-P, etc.)
│   ├── reasoning.py                 # Reasoning engine
│   ├── checkpoint_manager.py        # SafeTensors checkpoint management
│   ├── agi_system.py                # AGI system abstraction
│   ├── inference_engine.py          # Unified inference
│   ├── model_factory.py             # Model construction
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
│   │   ├── quantum_agi_core.py
│   │   └── quantum_readiness.py
│   │
│   ├── ethics/                      # Ethical AI
│   │   ├── feedback_collector.py
│   │   └── reward_model.py
│   │
│   ├── agi/                         # AGI sub-components
│   └── components/                  # Reusable components
│
├── modules/                         # Feature modules
│   ├── __init__.py
│   │
│   ├── multimodal/                  # Multimodal processing
│   │   ├── fusion_module.py
│   │   ├── hybrid_audio_module.py
│   │   └── hybrid_video_module.py
│   │
│   ├── tokenization/                # Tokenization
│   │   └── multimodal_tokenizer.py
│   │
│   ├── hybrid_architecture/         # Multi-paradigm learning
│   │   └── hybrid_integrator.py
│   │
│   ├── capabilities/                # AGI capabilities
│   │   ├── integrated_agi_system.py
│   │   ├── real_time_learning.py
│   │   └── zero_shot_reasoning.py
│   │
│   └── integrations/                # External integrations
│       └── web_integration.py
│
├── data_processing/                 # Data utilities
│   ├── data_processor.py
│   └── data_utils.py
│
├── image_generation/                # Image generation app
│   ├── api.py
│   ├── chat_ui.py
│   ├── model_module.py
│   └── train.py
│
├── text_summarization/              # Text summarization app
│   ├── api.py
│   ├── chat_ui.py
│   ├── text_summary_module.py
│   └── train.py
│
├── tests/                           # Test suite
│   ├── test_framework.py            # Main test framework
│   ├── test_runner.py               # Test orchestration
│   ├── test_tokenizer.py
│   ├── test_config.py
│   ├── system_validator.py
│   ├── test_data/                   # Test fixtures
│   ├── test_model/                  # Model tests
│   └── demo/                        # Demo scripts
│
├── docs/                            # Documentation
│   └── ARCHITECTURE.md              # This file
│
├── rtdlm.py                         # Main AGI model
├── train.py                         # Training script
├── inference.py                     # Inference & demonstration
├── train_tokenizer.py               # Tokenizer training
├── install_dependencies.py          # Dependency installer
├── requirements.txt                 # Python dependencies
├── LICENSE                          # License file
└── README.md                        # Project readme
```

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
│  │SentencePiece│  │   FAISS     │  │   NumPy     │             │
│  │             │  │             │  │             │             │
│  │ - BPE       │  │ - IndexFlat │  │ - Arrays    │             │
│  │ - Unigram   │  │ - Retrieval │  │ - Math ops  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  librosa    │  │  OpenCV     │  │ SafeTensors │             │
│  │             │  │             │  │             │             │
│  │ - Audio     │  │ - Video     │  │ - Secure    │             │
│  │ - MFCC      │  │ - Frames    │  │   weights   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
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
| Tokenization | `tokenization/` | 1 | Complete |
| Reasoning | `reasoning/` | 1 | Complete |
| Configuration | `config/` | 3 | Complete |
| Training | `train.py`, `train_*.py` | 4 | Complete |
| Inference | `inference.py` | 1 | Complete |
| Tests | `tests/` | 244 tests | Complete |

> **Note**: Data processing has been moved to [Auralith-Data-Pipeline](https://github.com/AuralithAI/Auralith-Data-Pipeline).
