from . import identify_objects, analyze_targets, refine_detections
from .manager import PromptManager

# Create a global prompt manager instance
prompt_manager = PromptManager()

__all__ = ["identify_objects", "analyze_targets", "refine_detections", "prompt_manager"]
