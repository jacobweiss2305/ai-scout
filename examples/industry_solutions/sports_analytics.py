from aiscout import Scout
from aiscout.providers.anthropic import LLM
import os
from datetime import datetime

# Get API key from environment
api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize LLM for automated labeling
llm = LLM(api_key=api_key, model="claude-3-7-sonnet-20250219")

"""
Sports Scene Labeling Example
---------------------------
This example demonstrates how to use AI-Scout to automate data labeling
for training YOLO models on sports imagery. AI-Scout combines YOLO's
base detection with LLM refinement to generate high-quality bounding
boxes, replacing manual labeling effort.

The output is an annotated image with bounding boxes that can be used
for training or fine-tuning YOLO models for sports applications.
"""

# Initialize Scout for automated labeling
scout = Scout(llm=llm, confidence_threshold=0.35, debug_mode=True)

# Create outputs directory for labeled images
output_dir = os.path.join(os.path.dirname(__file__), "labeled_data")
os.makedirs(output_dir, exist_ok=True)

# Example: Label a soccer match scene
image_path = os.path.join("..", "sample_images", "soccer-match.jpg")

# Define target objects for sports dataset
target_list = [
    "person",         # Players, referees, staff
    "sports ball",    # Soccer ball
    "goal post",      # Soccer goals
    "bench",          # Team benches
    "chair",          # Sideline seating
    "bottle",         # Water bottles
    "backpack",       # Player bags
    "uniform",        # Player uniforms
    "flag"           # Corner flags, linesmen flags
]

# Generate labels with bounding boxes
result = scout.detect(
    image_path,
    target_list=target_list,
    confidence_threshold=0.75,  # High threshold for quality labels
    min_iterations=4,          # Multiple refinement passes
    max_iterations=8
)

# Save labeled image for verification
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"labeled_{timestamp}.jpg")
result["annotated_image"].save(output_path)
print(f"\nSaved labeled image to: {output_path}")

# Print labeling results
print("\nGenerated Labels:")
print("-" * 30)
for detection in result["detections"]:
    print(f"Object: {detection['class']}")
    print(f"  Confidence: {detection['confidence']:.2%}")
    print(f"  Bounding Box: {detection['box']}")

print("\nNote: These automated labels can be used to train or fine-tune")
print("YOLO models for sports object detection. Review the labeled")
print("images to ensure quality before using them in your training dataset.")
