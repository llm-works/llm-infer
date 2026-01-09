"""Generation guards for quality control.

Guards are pluggable checks that run during generation to detect
issues like repetition, low confidence, or other quality problems.
"""

from .protocol import GenerationGuard, GuardResult
from .repetition import RepetitionGuard

__all__ = ["GenerationGuard", "GuardResult", "RepetitionGuard"]
