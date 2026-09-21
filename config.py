"""
Top-level Configuration Re-export.

Provides direct access to PipelineConfig and DEFAULT_CONFIG from src.config.
"""

from src.config import PipelineConfig, DEFAULT_CONFIG

__all__ = ["PipelineConfig", "DEFAULT_CONFIG"]

