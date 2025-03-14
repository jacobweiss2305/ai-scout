import os
from datetime import datetime
from aiscout import Scout
from aiscout.providers.anthropic import LLM

# Get API key from environment
api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize LLM
llm = LLM(api_key=api_key, model="claude-3-7-sonnet-20250219")

# Initialize detector with debug mode
scout = Scout(llm=llm, debug_mode=True)

# Create outputs directory
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Run detection
image_path = os.path.join("..", "sample_images", "soccer.png")
result = scout.detect(
    image_path,
    target_list=["players"],
    confidence_threshold=0.2,
    min_iterations=3,
    max_iterations=6
)

# Save annotated image
output_path = os.path.join(output_dir, "soccer_annotated.png")
result["annotated_image"].save(output_path)
print(f"\nSaved annotated image to: {output_path}")