# AI Scout

AI Scout is a flexible object detection system that combines YOLO with LLMs for intelligent object identification and analysis.

## Quick Start

```python
import os
from aiscout import Scout
from aiscout.providers.anthropic import LLM

# Initialize with your preferred LLM
api_key = os.getenv("ANTHROPIC_API_KEY")
llm = LLM(api_key=api_key, model="claude-3-7-sonnet-20250219")
scout = Scout(llm=llm)

# Run detection
result = scout.detect(
    "path/to/image.jpg",
    target_list=["target1", "target2"],
    confidence_threshold=0.2
)

# Save annotated image
result["annotated_image"].save("output.jpg")
```

## Features

- Combines YOLO and LLM capabilities for enhanced object detection
- Supports multiple LLM providers (Anthropic Claude, OpenAI GPT-4V)
- Advanced prompt customization and management
- Iterative refinement with configurable iterations
- Debug mode for development
- Flexible target specification
- Provider-agnostic interface

## Requirements

- Python >=3.9
- ultralytics (YOLO)
- requests
- rich
- Anthropic API key (for Claude) or OpenAI API key (for GPT-4V)

## Installation

```bash
pip install aiscout
```

## Configuration

Set your API key as an environment variable:
```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your_api_key"
# For OpenAI
export OPENAI_API_KEY="your_api_key"
```

## Advanced Usage

```python
# Enable debug mode
scout = Scout(llm=llm, debug_mode=True)

# Configure detection parameters
result = scout.detect(
    "image.jpg",
    target_list=["target1", "target2"],
    confidence_threshold=0.2,
    min_iterations=3,
    max_iterations=6
)
```

## Prompt Customization

```python
from aiscout.prompts import prompt_manager

# Replace entire prompt
prompt_manager.set_prompt(
    "identify_objects",
    """Analyze this image and identify objects with these requirements:
1. Focus on vehicles and traffic signs
2. Identify make and model when possible
3. Note any safety hazards"""
)

# Append additional instructions
prompt_manager.append_to_prompt(
    "analyze_targets",
    "Additional requirement: Prioritize specific vehicle types over generic classes"
)

# Reset prompts
prompt_manager.reset_prompt("identify_objects")  # Reset specific prompt
prompt_manager.reset_all()  # Reset all prompts
```

Available prompt types:
- `identify_objects`: Initial object identification
- `analyze_targets`: Target analysis and mapping
- `refine_detections`: Detection refinement rules

## Examples

The `examples` directory contains:
- `anthropic/`: Claude integration example
- `openai/`: GPT-4V integration example
- `custom_prompts/`: Prompt customization examples
- `sample_images/`: Test images

## License

MIT License
