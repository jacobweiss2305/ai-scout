from aiscout import Scout
from aiscout.providers.anthropic import LLM
from aiscout.prompts import prompt_manager

# Initialize LLM with your API key
llm = LLM(api_key="your_api_key")

# Example 1: Replace the entire identify_objects prompt
prompt_manager.set_prompt(
    "identify_objects",
    """Analyze this image and identify objects with these specific requirements:
1. Focus on vehicles and traffic signs
2. Identify make and model of vehicles when possible
3. Note any safety hazards or violations

Return a JSON with this structure:
{
  "identified_objects": [
    {
      "label": "specific object name",
      "priority": 1-5,
      "rationale": "brief explanation",
      "possible_subclasses": ["subcategory1", "subcategory2"]
    }
  ],
  "detected_scene_type": "brief description"
}"""
)

# Example 2: Append additional instructions to analyze_targets
prompt_manager.append_to_prompt(
    "analyze_targets",
    """Additional requirements:
1. For vehicles, prioritize matching to specific vehicle types over generic 'car' class
2. For traffic signs, ensure high confidence thresholds (>0.8)
3. Pay special attention to pedestrians and cyclists"""
)

# Example 3: Multiple appends to refine_detections
prompt_manager.append_to_prompt(
    "refine_detections",
    "When refining vehicle detections, ensure bounding boxes include side mirrors"
)
prompt_manager.append_to_prompt(
    "refine_detections",
    "For traffic signs, prefer tighter bounding boxes to reduce false positives"
)

# Initialize Scout with customized prompts
scout = Scout(llm=llm)

# Run detection - it will use your customized prompts
detections = scout.detect("path/to/your/image.jpg")

# Reset a specific prompt to default
prompt_manager.reset_prompt("identify_objects")

# Reset all prompts to defaults
prompt_manager.reset_all()
