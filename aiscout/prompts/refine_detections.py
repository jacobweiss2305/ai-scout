PROMPT = """I need you to refine these object detections based on the image and the target list.

Target objects: {target_list}
Image dimensions: Width={image_width}, Height={image_height}
YOLO detections: {detections_json}
Potentially contained boxes (indexes): {contained_boxes_json}
Overlapping boxes (indexes and IoU): {overlapping_boxes_json}
Current iteration: {iteration}
Previous iteration results: {previous_json}

Your task:
1. Review the YOLO detections and the image
2. Analyze each detection and assess if it's correct
3. You have FULL AUTHORITY to:
   - REMOVE false positive detections entirely
   - ADJUST confidence scores up or down based on visual assessment
   - RELABEL objects to better match what they actually are
   - RESOLVE overlapping or contained detections
4. For contained detections, determine if they are:
   - Valid sub-objects
   - False positives
5. For overlapping detections, determine if they are:
   - The same object detected multiple times
   - Different objects that are close to each other
6. Ensure consistent labeling
7. Focus on accuracy over quantity - it's better to have fewer correct detections than many questionable ones
8. If this is not the first iteration, review the previous results and:
   - Note what has improved
   - Identify remaining issues to fix
   - Determine if further iterations are needed
9. Ensure the bounding box fully contains the object
10. If a bounding box is not fully contained, you can adjust it to be more accurate OR remove the detection entirely

Return a JSON object with this exact structure:
{{
  "refined_detections": [
    {{
      "box": [x1, y1, x2, y2],
      "class": "object class",
      "confidence": 0.95,
      "notes": "explanation of any changes made"
    }}
  ],
  "removed_detections": [
    {{
      "box": [x1, y1, x2, y2],
      "class": "object class",
      "confidence": 0.95,
      "reason": "why this detection was removed"
    }}
  ],
  "analysis": {{
    "improvements": ["list of improvements made"],
    "remaining_issues": ["list of issues still to address"],
    "confidence_score": 0.95,
    "continue_iteration": true
  }}
}}

Only respond with the JSON object, no other text. Your response must start with '{{' and end with '}}'."""
