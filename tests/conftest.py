"""pytest configuration: add project src/ to the Python path."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
