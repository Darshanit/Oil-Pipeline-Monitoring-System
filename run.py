#!/usr/bin/env python3
"""
Oil Pipeline Pressure Monitoring & Leak Detection System - Main Entry Point

Usage:
  python run.py
"""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.run_detection import run_full_pipeline


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_full_pipeline(base_dir)