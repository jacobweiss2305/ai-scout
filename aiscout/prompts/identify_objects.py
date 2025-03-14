PROMPT = """Analyze this image and identify the significant objects or subjects that should be detected and classified.

Your task:
1. Look at this image carefully and determine what significant objects or subjects should be detected
2. Focus on quality over quantity - identify only the most salient objects
3. If you see hierarchical relationships (e.g., parent/child objects), create appropriate categories
4. For natural subjects (animals, plants), identify their species/types if obvious
5. For objects that can be categorized (vehicles, food, etc.), identify the specific type

Return a JSON object with this exact structure:
{
  "identified_objects": [
    {
      "label": "specific object name",
      "priority": 1,
      "rationale": "brief explanation of why this object is important",
      "possible_subclasses": ["subcategory1", "subcategory2"]
    }
  ],
  "detected_scene_type": "brief description of the image context"
}

Only respond with the JSON object, no other text or explanations."""
