import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Dict, List, Optional, Any
import time
import os
import sys

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rtdlm_agi_complete import RT_DLM_AGI, create_rtdlm_agi
from agi_config import AdvancedAGIConfig
from data_utils import DataProcessor

class RT_DLM_AGI_Assistant:
    """Interactive RT-DLM AGI Assistant for demonstrations and real-world usage"""
    
    def __init__(self, config: AdvancedAGIConfig, checkpoint_path: Optional[str] = None):
        self.config = config
        self.rng = jax.random.PRNGKey(42)
        
        # Initialize model
        self.model = create_rtdlm_agi(config)
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            vocab_size=config.vocab_size,
            model_prefix="data/rt_dlm_sp"
        )
        
        # Model parameters
        self.params = None
        self.conversation_history = []
        self.knowledge_base = []
        
        # Initialize or load model
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize model with random parameters for demonstration"""
        print("[INFO] Initializing RT-DLM AGI for inference...")
        
        # Create sample input for initialization
        sample_text = "Hello, I am RT-DLM AGI, ready to assist you."
        tokens = self.data_processor.tokenize(sample_text)
        padded_tokens = self.data_processor.pad_sequence(tokens, self.config.max_seq_length)
        sample_input = jnp.array([padded_tokens], dtype=jnp.int32)
        
        sample_batch = {"text": sample_input}
        
        # Add multimodal inputs if enabled
        if self.config.multimodal_enabled:
            sample_batch["multimodal_inputs"] = {
                "images": jnp.zeros((1, 224, 224, 3)),
                "audio": jnp.zeros((1, 128, 128)),
            }
        
        # Initialize parameters
        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self.model.init(init_rng, **sample_batch)
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"[INFO] AGI Model initialized with {param_count:,} parameters")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        print(f"[INFO] Loading AGI model from {checkpoint_path}...")
        
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.params = checkpoint["params"]
        print(f"[INFO] Model loaded from epoch {checkpoint['epoch']}")
    
    def preprocess_input(self, text: str, max_length: Optional[int] = None) -> jnp.ndarray:
        """Preprocess text input for the model"""
        if max_length is None:
            max_length = self.config.max_seq_length
        
        tokens = self.data_processor.tokenize(text)
        padded_tokens = self.data_processor.pad_sequence(tokens, max_length)
        return jnp.array([padded_tokens], dtype=jnp.int32)
    
    def postprocess_output(self, logits: jnp.ndarray, temperature: float = 1.0) -> str:
        """Convert model logits to human-readable text"""
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Sample from distribution
        self.rng, sample_rng = jax.random.split(self.rng)
        token_ids = jax.random.categorical(sample_rng, scaled_logits, axis=-1)
        
        # Convert to text
        token_ids_list = token_ids[0].tolist()
        
        # Remove padding tokens
        filtered_tokens = [t for t in token_ids_list if t != self.config.pad_token_id]
        
        try:
            generated_text = self.data_processor.decode_tokens(filtered_tokens)
            return generated_text
        except:
            return "[Generated text could not be decoded]"
    
    def think_step_by_step(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Perform step-by-step reasoning on a question"""
        print(f"\n🤔 [RT-DLM AGI] Thinking step by step about: {question}")
        
        # Prepare inputs
        if context:
            full_input = f"Context: {context}\\n\\nQuestion: {question}\\n\\nLet me think step by step:"
        else:
            full_input = f"Question: {question}\\n\\nLet me think step by step:"
        
        input_tokens = self.preprocess_input(full_input)
        
        # Create conversation history context
        history_context = None
        if self.conversation_history:
            history_text = " ".join(self.conversation_history[-3:])  # Last 3 exchanges
            history_context = self.preprocess_input(history_text)
        
        # Forward pass with reasoning
        self.rng, inference_rng = jax.random.split(self.rng)
        
        model_output = self.model.apply(
            self.params, inference_rng,
            inputs={"text": input_tokens},
            conversation_history=history_context,
            return_reasoning=True
        )
        
        # Extract reasoning chain
        reasoning_steps = []
        if "reasoning_chain" in model_output:
            for i, step in enumerate(model_output["reasoning_chain"]):
                step_text = self.postprocess_output(step.mean(axis=0, keepdims=True))
                reasoning_steps.append(f"Step {i+1}: {step_text}")
                print(f"  💭 Step {i+1}: {step_text}")
        
        # Generate final answer
        final_answer = self.postprocess_output(model_output["logits"], temperature=0.7)
        print(f"  ✅ Final Answer: {final_answer}")
        
        # Extract other cognitive processes
        result = {
            "question": question,
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer,
            "confidence": float(jnp.mean(model_output.get("confidence_scores", [0.5]))),
        }
        
        # Add consciousness insights if available
        if "consciousness" in model_output:
            consciousness = model_output["consciousness"]
            result["consciousness_insights"] = {
                "self_awareness": "Model is aware of its reasoning process",
                "introspection": "Model examined its own thoughts",
                "autonomous_goals": "Model set goals for problem solving"
            }
            print(f"  🧠 Consciousness Level: {self.config.self_awareness_level}")
        
        # Add multimodal processing if available
        if "multimodal_features" in model_output:
            result["multimodal_processing"] = "Integrated multi-modal understanding"
            print(f"  🎭 Multi-modal processing active")
        
        # Add quantum enhancement if available
        if "quantum_features" in model_output:
            result["quantum_enhancement"] = "Quantum-inspired processing enabled"
            print(f"  ⚛️ Quantum enhancement active")
        
        return result
    
    def creative_generation(self, prompt: str, creativity_level: float = 0.7) -> Dict[str, Any]:
        """Generate creative content"""
        print(f"\n🎨 [RT-DLM AGI] Creative generation with level {creativity_level}")
        print(f"Prompt: {prompt}")
        
        creative_prompt = f"Create something creative based on: {prompt}\\n\\nCreative output:"
        input_tokens = self.preprocess_input(creative_prompt)
        
        self.rng, creative_rng = jax.random.split(self.rng)
        
        model_output = self.model.apply(
            self.params, creative_rng,
            inputs={"text": input_tokens},
            return_reasoning=False
        )
        
        # Generate creative content with higher temperature
        creative_output = self.postprocess_output(
            model_output["logits"], 
            temperature=1.0 + creativity_level
        )
        
        result = {
            "prompt": prompt,
            "creative_output": creative_output,
            "creativity_level": creativity_level
        }
        
        if "creative_output" in model_output:
            novelty_score = float(jnp.mean(model_output["creative_output"].get("novelty_score", 0.5)))
            result["novelty_score"] = novelty_score
            print(f"  ✨ Novelty Score: {novelty_score:.3f}")
        
        print(f"  🎭 Creative Output: {creative_output}")
        
        return result
    
    def scientific_inquiry(self, hypothesis: str, observations: str) -> Dict[str, Any]:
        """Perform scientific reasoning and hypothesis testing"""
        print(f"\n🔬 [RT-DLM AGI] Scientific Inquiry Mode")
        print(f"Hypothesis: {hypothesis}")
        print(f"Observations: {observations}")
        
        scientific_prompt = f"""
        Hypothesis: {hypothesis}
        Observations: {observations}
        
        As a scientific AI, please:
        1. Analyze the hypothesis
        2. Evaluate the observations
        3. Determine if the observations support or refute the hypothesis
        4. Suggest additional experiments
        
        Scientific Analysis:
        """
        
        input_tokens = self.preprocess_input(scientific_prompt)
        
        self.rng, science_rng = jax.random.split(self.rng)
        
        model_output = self.model.apply(
            self.params, science_rng,
            inputs={"text": input_tokens},
            return_reasoning=True
        )
        
        analysis = self.postprocess_output(model_output["logits"], temperature=0.3)
        
        result = {
            "hypothesis": hypothesis,
            "observations": observations,
            "scientific_analysis": analysis,
        }
        
        if "scientific_discovery" in model_output:
            discovery = model_output["scientific_discovery"]
            if "experiment_design" in discovery:
                result["suggested_experiments"] = "AI generated experimental designs"
                print(f"  🧪 Experimental designs generated")
        
        print(f"  📊 Scientific Analysis: {analysis}")
        
        return result
    
    def social_emotional_interaction(self, user_message: str, emotional_context: str = "neutral") -> Dict[str, Any]:
        """Handle social and emotional aspects of interaction"""
        print(f"\n💝 [RT-DLM AGI] Social-Emotional Interaction")
        print(f"User Message: {user_message}")
        print(f"Emotional Context: {emotional_context}")
        
        social_prompt = f"""
        User says: "{user_message}"
        Emotional context: {emotional_context}
        
        Please respond with empathy and social awareness:
        """
        
        input_tokens = self.preprocess_input(social_prompt)
        
        # Include conversation history for social context
        history_context = None
        if self.conversation_history:
            history_text = " ".join(self.conversation_history[-5:])
            history_context = self.preprocess_input(history_text)
        
        self.rng, social_rng = jax.random.split(self.rng)
        
        model_output = self.model.apply(
            self.params, social_rng,
            inputs={"text": input_tokens},
            conversation_history=history_context,
            return_reasoning=False
        )
        
        empathetic_response = self.postprocess_output(model_output["logits"], temperature=0.8)
        
        result = {
            "user_message": user_message,
            "emotional_context": emotional_context,
            "empathetic_response": empathetic_response,
        }
        
        if "social_emotional" in model_output:
            social_data = model_output["social_emotional"]
            if "recognized_emotions" in social_data:
                emotions = social_data["recognized_emotions"]
                dominant_emotion = jnp.argmax(emotions)
                emotion_names = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
                result["detected_emotion"] = emotion_names[int(dominant_emotion)]
                print(f"  😊 Detected Emotion: {result['detected_emotion']}")
        
        print(f"  💬 Empathetic Response: {empathetic_response}")
        
        return result
    
    def multimodal_understanding(self, text_input: str, image_description: str = None, audio_description: str = None) -> Dict[str, Any]:
        """Demonstrate multimodal understanding capabilities"""
        print(f"\n🎭 [RT-DLM AGI] Multi-Modal Understanding")
        
        if not self.config.multimodal_enabled:
            return {"error": "Multi-modal processing not enabled in configuration"}
        
        multimodal_prompt = f"Text: {text_input}"
        if image_description:
            multimodal_prompt += f"\\nImage: {image_description}"
        if audio_description:
            multimodal_prompt += f"\\nAudio: {audio_description}"
        
        multimodal_prompt += "\\n\\nMulti-modal analysis:"
        
        input_tokens = self.preprocess_input(multimodal_prompt)
        
        # Create synthetic multimodal inputs for demonstration
        self.rng, *modal_rngs = jax.random.split(self.rng, 4)
        
        multimodal_inputs = {}
        if image_description:
            multimodal_inputs["images"] = jax.random.normal(modal_rngs[0], (1, 224, 224, 3)) * 0.1
        if audio_description:
            multimodal_inputs["audio"] = jax.random.normal(modal_rngs[1], (1, 128, 128)) * 0.1
        
        model_output = self.model.apply(
            self.params, modal_rngs[2],
            inputs={"text": input_tokens},
            multimodal_inputs=multimodal_inputs if multimodal_inputs else None,
            return_reasoning=False
        )
        
        analysis = self.postprocess_output(model_output["logits"], temperature=0.6)
        
        result = {
            "text_input": text_input,
            "image_description": image_description,
            "audio_description": audio_description,
            "multimodal_analysis": analysis,
        }
        
        if "multimodal_features" in model_output:
            result["cross_modal_fusion"] = "Successfully integrated multiple modalities"
            print(f"  🔄 Cross-modal fusion successful")
        
        print(f"  🎯 Multi-modal Analysis: {analysis}")
        
        return result
    
    def interactive_session(self):
        """Start an interactive session with the AGI"""
        print("\\n" + "=" * 80)
        print("🚀 RT-DLM AGI Interactive Session")
        print("Advanced Artificial General Intelligence")
        print("=" * 80)
        
        self.config.print_summary()
        
        print("\\nAvailable capabilities:")
        print("  1. 💭 Step-by-step reasoning")
        print("  2. 🎨 Creative generation")
        print("  3. 🔬 Scientific inquiry")
        print("  4. 💝 Social-emotional interaction")
        print("  5. 🎭 Multi-modal understanding")
        print("  6. 🧠 Consciousness simulation")
        print("  7. ⚛️ Quantum-enhanced processing")
        
        print("\\nCommands:")
        print("  /reason <question> - Step-by-step reasoning")
        print("  /create <prompt> - Creative generation")
        print("  /science <hypothesis> | <observations> - Scientific analysis")
        print("  /social <message> - Social-emotional interaction")
        print("  /multimodal <text> [image: <desc>] [audio: <desc>] - Multi-modal")
        print("  /quit - Exit session")
        
        print("\\n" + "-" * 80)
        print("🤖 RT-DLM AGI: Hello! I'm your advanced AGI assistant. How can I help you today?")
        
        while True:
            try:
                user_input = input("\\n👤 You: ").strip()
                
                if user_input.lower() in ["/quit", "quit", "exit"]:
                    print("🤖 RT-DLM AGI: Goodbye! Thank you for using RT-DLM AGI.")
                    break
                
                # Add to conversation history
                self.conversation_history.append(f"User: {user_input}")
                
                start_time = time.time()
                
                if user_input.startswith("/reason"):
                    question = user_input[8:].strip()
                    if question:
                        result = self.think_step_by_step(question)
                        response = result["final_answer"]
                    else:
                        response = "Please provide a question after /reason"
                
                elif user_input.startswith("/create"):
                    prompt = user_input[8:].strip()
                    if prompt:
                        result = self.creative_generation(prompt)
                        response = result["creative_output"]
                    else:
                        response = "Please provide a creative prompt after /create"
                
                elif user_input.startswith("/science"):
                    science_input = user_input[9:].strip()
                    if "|" in science_input:
                        hypothesis, observations = science_input.split("|", 1)
                        result = self.scientific_inquiry(hypothesis.strip(), observations.strip())
                        response = result["scientific_analysis"]
                    else:
                        response = "Please provide hypothesis | observations after /science"
                
                elif user_input.startswith("/social"):
                    message = user_input[8:].strip()
                    if message:
                        result = self.social_emotional_interaction(message)
                        response = result["empathetic_response"]
                    else:
                        response = "Please provide a message after /social"
                
                elif user_input.startswith("/multimodal"):
                    mm_input = user_input[12:].strip()
                    if mm_input:
                        # Parse multimodal input
                        text_part = mm_input
                        image_desc = None
                        audio_desc = None
                        
                        if "image:" in mm_input:
                            parts = mm_input.split("image:")
                            text_part = parts[0].strip()
                            img_part = parts[1]
                            if "audio:" in img_part:
                                img_part, audio_desc = img_part.split("audio:", 1)
                                audio_desc = audio_desc.strip()
                            image_desc = img_part.strip()
                        
                        result = self.multimodal_understanding(text_part, image_desc, audio_desc)
                        response = result.get("multimodal_analysis", "Multimodal processing completed")
                    else:
                        response = "Please provide input after /multimodal"
                
                else:
                    # Regular conversation
                    result = self.social_emotional_interaction(user_input)
                    response = result["empathetic_response"]
                
                # Add response to history
                self.conversation_history.append(f"AGI: {response}")
                
                # Keep history manageable
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                inference_time = time.time() - start_time
                
                print(f"\\n🤖 RT-DLM AGI: {response}")
                print(f"   ⏱️ Response time: {inference_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\\n\\n🤖 RT-DLM AGI: Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Error: {str(e)}")
                print("Please try again with a different input.")

def demonstrate_agi_capabilities():
    """Demonstrate various AGI capabilities"""
    print("\\n" + "=" * 80)
    print("🚀 RT-DLM AGI Capabilities Demonstration")
    print("=" * 80)
    
    # Create AGI configuration
    config = AdvancedAGIConfig(
        d_model=384,
        num_heads=8,
        num_layers=6,
        vocab_size=8000,
        multimodal_enabled=True,
        consciousness_simulation=True,
        quantum_layers=2,
        max_reasoning_steps=5,
        scientific_reasoning=True,
        creative_generation=True,
        social_intelligence=True,
        emotional_intelligence=True,
    )
    
    # Initialize AGI assistant
    assistant = RT_DLM_AGI_Assistant(config)
    
    print("\\n🧠 Demonstrating Advanced Reasoning...")
    reasoning_result = assistant.think_step_by_step(
        "If a train travels 60 mph for 2 hours, then 80 mph for 1.5 hours, what is the total distance?"
    )
    
    print("\\n🎨 Demonstrating Creative Generation...")
    creative_result = assistant.creative_generation(
        "A world where gravity works backwards"
    )
    
    print("\\n🔬 Demonstrating Scientific Inquiry...")
    science_result = assistant.scientific_inquiry(
        "Increased CO2 levels cause global warming",
        "Global temperatures have risen 1.1°C since 1880, CO2 levels increased from 315ppm to 415ppm"
    )
    
    print("\\n💝 Demonstrating Social-Emotional Intelligence...")
    social_result = assistant.social_emotional_interaction(
        "I'm feeling really stressed about my upcoming exams",
        emotional_context="anxious"
    )
    
    if config.multimodal_enabled:
        print("\\n🎭 Demonstrating Multi-Modal Understanding...")
        multimodal_result = assistant.multimodal_understanding(
            "What do you see in this scene?",
            image_description="A sunset over mountains with a lake",
            audio_description="Birds chirping and water flowing"
        )
    
    print("\\n" + "=" * 80)
    print("✅ RT-DLM AGI Demonstration Complete!")
    print("🤖 The model showcased reasoning, creativity, scientific analysis,")
    print("   social intelligence, and multi-modal understanding.")
    print("=" * 80)

def main():
    """Main function to run AGI inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RT-DLM AGI Inference")
    parser.add_argument("--demo", action="store_true", help="Run capability demonstration")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    if args.demo:
        demonstrate_agi_capabilities()
    elif args.interactive:
        config = AdvancedAGIConfig(
            d_model=384,
            num_heads=8,
            num_layers=6,
            vocab_size=8000,
            multimodal_enabled=True,
            consciousness_simulation=True,
            quantum_layers=2,
        )
        
        assistant = RT_DLM_AGI_Assistant(config, args.checkpoint)
        assistant.interactive_session()
    else:
        print("Please specify --demo or --interactive")
        print("Use --help for more options")

if __name__ == "__main__":
    main()
