"""
Shape Validation Tests for RT-DLM AGI System

This test suite validates shape compatibility across all module integration points
to catch dimension mismatches early in development.

Test Categories:
- test_tms_shapes.py: TMSModel shapes with batch/seq/d_model variations
- test_agi_system_shapes.py: RTDLMAGISystem module chain shapes
- test_module_integration_shapes.py: Individual module boundary shapes

Coverage:
- Batch size variations (1, 4, 16, 32)
- Sequence length variations (8, 16, 32, 64, 128)
- d_model variations (32, 64, 128, 256)
- Memory integration (LTM, STM, MTM)
- Multimodal inputs (audio, video, image)
- Module boundary compatibility

Run tests:
    pytest tests/shape_validation/ -v
"""
