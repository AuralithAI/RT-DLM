"""
Tests for Integrated AGI System Module

Tests for the complete AGI system integration including memory management,
task processing, and capability coordination.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
import time


class TestAGIStage(unittest.TestCase):
    """Test AGIStage enum."""
    
    def test_agi_stages_exist(self):
        """Test all AGI stages are defined."""
        from src.modules.capabilities.integrated_agi_system import AGIStage
        
        self.assertEqual(AGIStage.STAGE_0_MVP.value, "Stage 0: Minimum Viable Product")
        self.assertEqual(AGIStage.STAGE_1_FOUNDATION.value, "Stage 1: Foundation AGI")
        self.assertEqual(AGIStage.STAGE_2_SOPHISTICATED.value, "Stage 2: Sophisticated AGI")
        self.assertEqual(AGIStage.STAGE_3_ADVANCED.value, "Stage 3: Advanced AGI")


class TestTaskType(unittest.TestCase):
    """Test TaskType enum."""
    
    def test_task_types_exist(self):
        """Test all task types are defined."""
        from src.modules.capabilities.integrated_agi_system import TaskType
        
        self.assertEqual(TaskType.REASONING.value, "reasoning")
        self.assertEqual(TaskType.LEARNING.value, "learning")
        self.assertEqual(TaskType.CREATION.value, "creation")
        self.assertEqual(TaskType.PROBLEM_SOLVING.value, "problem_solving")
        self.assertEqual(TaskType.ANALYSIS.value, "analysis")


class TestAGICapability(unittest.TestCase):
    """Test AGICapability dataclass."""
    
    def test_capability_creation(self):
        """Test creating AGI capabilities."""
        from src.modules.capabilities.integrated_agi_system import (
            AGICapability, AGIStage
        )
        
        capability = AGICapability(
            name="reasoning",
            description="Basic reasoning capability",
            stage_required=AGIStage.STAGE_1_FOUNDATION,
            confidence_level=0.85,
            is_active=True,
            dependencies=["memory", "attention"],
            performance_metrics={"accuracy": 0.9, "latency": 0.1}
        )
        
        self.assertEqual(capability.name, "reasoning")
        self.assertEqual(capability.confidence_level, 0.85)
        self.assertTrue(capability.is_active)
        self.assertEqual(len(capability.dependencies), 2)


class TestTaskRequest(unittest.TestCase):
    """Test TaskRequest dataclass."""
    
    def test_task_request_creation(self):
        """Test creating task requests."""
        from src.modules.capabilities.integrated_agi_system import (
            TaskRequest, TaskType
        )
        
        request = TaskRequest(
            task_id="task_001",
            task_type=TaskType.REASONING,
            description="Solve a logic puzzle",
            context={"puzzle": "example"},
            priority=8
        )
        
        self.assertEqual(request.task_id, "task_001")
        self.assertEqual(request.task_type, TaskType.REASONING)
        self.assertEqual(request.priority, 8)
        self.assertEqual(request.user_id, "default")
    
    def test_task_request_defaults(self):
        """Test task request default values."""
        from src.modules.capabilities.integrated_agi_system import (
            TaskRequest, TaskType
        )
        
        request = TaskRequest(
            task_id="task_002",
            task_type=TaskType.ANALYSIS,
            description="Analyze data",
            context={}
        )
        
        self.assertEqual(request.priority, 5)
        self.assertIsNone(request.deadline)


class TestTaskResponse(unittest.TestCase):
    """Test TaskResponse dataclass."""
    
    def test_task_response_creation(self):
        """Test creating task responses."""
        from src.modules.capabilities.integrated_agi_system import (
            TaskResponse, AGIStage
        )
        
        response = TaskResponse(
            task_id="task_001",
            success=True,
            result={"answer": 42},
            reasoning_chain=None,
            confidence=0.95,
            execution_time=0.5,
            stage_used=AGIStage.STAGE_2_SOPHISTICATED,
            learned_from_task=True
        )
        
        self.assertEqual(response.task_id, "task_001")
        self.assertTrue(response.success)
        self.assertEqual(response.confidence, 0.95)
        self.assertTrue(response.learned_from_task)


class TestAGIMemoryManager(unittest.TestCase):
    """Test AGIMemoryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.modules.capabilities.integrated_agi_system import AGIMemoryManager
        self.manager = AGIMemoryManager(d_model=64)
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        self.assertEqual(self.manager.d_model, 64)
        self.assertEqual(len(self.manager.episodic_memory), 0)
        self.assertEqual(len(self.manager.semantic_memory), 0)
        self.assertEqual(len(self.manager.procedural_memory), 0)
    
    def test_store_episode(self):
        """Test storing task episodes."""
        from src.modules.capabilities.integrated_agi_system import (
            TaskRequest, TaskResponse, TaskType, AGIStage
        )
        
        request = TaskRequest(
            task_id="test_001",
            task_type=TaskType.REASONING,
            description="Test task",
            context={"key": "value"}
        )
        
        response = TaskResponse(
            task_id="test_001",
            success=True,
            result="test result",
            reasoning_chain=None,
            confidence=0.9,
            execution_time=0.1,
            stage_used=AGIStage.STAGE_1_FOUNDATION,
            learned_from_task=True
        )
        
        self.manager.store_episode(request, response)
        
        self.assertEqual(len(self.manager.episodic_memory), 1)
        episode = self.manager.episodic_memory[0]
        self.assertEqual(episode["task_request"], request)
        self.assertEqual(episode["task_response"], response)
        self.assertTrue(episode["success"])
    
    def test_retrieve_relevant_episodes(self):
        """Test retrieving relevant episodes."""
        from src.modules.capabilities.integrated_agi_system import (
            TaskRequest, TaskResponse, TaskType, AGIStage
        )
        
        # Store some episodes
        for i in range(5):
            request = TaskRequest(
                task_id=f"task_{i}",
                task_type=TaskType.REASONING,
                description=f"Task {i}",
                context={"index": i}
            )
            response = TaskResponse(
                task_id=f"task_{i}",
                success=True,
                result=f"result_{i}",
                reasoning_chain=None,
                confidence=0.8 + i * 0.02,
                execution_time=0.1,
                stage_used=AGIStage.STAGE_1_FOUNDATION,
                learned_from_task=False
            )
            self.manager.store_episode(request, response)
        
        # Query for relevant episodes
        query = TaskRequest(
            task_id="query",
            task_type=TaskType.REASONING,
            description="Query task",
            context={"index": 2}
        )
        
        relevant = self.manager.retrieve_relevant_episodes(query, top_k=3)
        
        self.assertLessEqual(len(relevant), 3)
    
    def test_retrieve_from_empty_memory(self):
        """Test retrieving from empty memory."""
        from src.modules.capabilities.integrated_agi_system import (
            TaskRequest, TaskType
        )
        
        query = TaskRequest(
            task_id="query",
            task_type=TaskType.ANALYSIS,
            description="Test",
            context={}
        )
        
        relevant = self.manager.retrieve_relevant_episodes(query)
        self.assertEqual(len(relevant), 0)


class TestMemoryContextEncoding(unittest.TestCase):
    """Test memory context encoding."""
    
    def test_encode_context(self):
        """Test context encoding produces correct shape."""
        from src.modules.capabilities.integrated_agi_system import AGIMemoryManager
        
        manager = AGIMemoryManager(d_model=128)
        
        context = {"key1": "value1", "key2": [1, 2, 3]}
        embedding = manager._encode_context(context)
        
        self.assertEqual(embedding.shape, (128,))


class TestIntegratedAGIWorkflow(unittest.TestCase):
    """Integration tests for AGI workflow."""
    
    def test_full_task_workflow(self):
        """Test complete task submission and response workflow."""
        from src.modules.capabilities.integrated_agi_system import (
            AGIMemoryManager, TaskRequest, TaskResponse, TaskType, AGIStage
        )
        
        manager = AGIMemoryManager(d_model=64)
        
        # Create and process task
        request = TaskRequest(
            task_id="workflow_test",
            task_type=TaskType.PROBLEM_SOLVING,
            description="Solve a complex problem",
            context={"problem": "data", "constraints": ["time", "memory"]}
        )
        
        # Simulate processing
        response = TaskResponse(
            task_id="workflow_test",
            success=True,
            result={"solution": "found"},
            reasoning_chain=None,
            confidence=0.85,
            execution_time=1.5,
            stage_used=AGIStage.STAGE_2_SOPHISTICATED,
            learned_from_task=True
        )
        
        # Store in memory
        manager.store_episode(request, response)
        
        # Verify memory was updated
        self.assertEqual(len(manager.episodic_memory), 1)
        
        # Query for similar tasks
        similar_request = TaskRequest(
            task_id="similar",
            task_type=TaskType.PROBLEM_SOLVING,
            description="Another problem",
            context={"problem": "similar_data"}
        )
        
        relevant = manager.retrieve_relevant_episodes(similar_request, top_k=1)
        self.assertEqual(len(relevant), 1)


if __name__ == "__main__":
    unittest.main()
