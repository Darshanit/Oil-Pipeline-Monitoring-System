import os
import sys

# Ensure root directory is in sys.path during pytest run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
