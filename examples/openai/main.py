from aiscout import Scout
from aiscout.providers.openai import LLM

# Pick you LLM
llm = LLM(api_key="your_api_key", model="gpt-4o-mini")

# Initialize detector
scout = Scout(llm=llm)

# Run detection
image_path = "path/to/your/image.jpg"
detections = scout.detect(image_path)

# Print results
print(detections)