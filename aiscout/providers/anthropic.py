import json
import requests
from .base import BaseProvider
from ..prompts.identify_objects import PROMPT as IDENTIFY_OBJECTS_PROMPT
from ..prompts.analyze_targets import PROMPT as ANALYZE_TARGETS_PROMPT
from ..prompts.refine_detections import PROMPT as REFINE_DETECTIONS_PROMPT
from ..prompts import prompt_manager
import logging

logger = logging.getLogger(__name__)

class LLM(BaseProvider):
    def __init__(self, api_key, model="claude-3-7-sonnet-20250219"):
        super().__init__(api_key, model)
        
    def identify_objects(self, image_base64):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._encode_image(image_base64)
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_manager.get_prompt("identify_objects", IDENTIFY_OBJECTS_PROMPT)
                    }
                ]
            }
        ]
        
        response = self._make_llm_request(prompt, max_tokens=1500)
        try:
            content = response["content"][0]["text"]
            # Extract just the JSON part from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                objects_info = json.loads(json_content)
                
                # Extract target list from identified objects
                target_list = []
                if "identified_objects" in objects_info:
                    for obj in objects_info["identified_objects"]:
                        if "label" in obj:
                            target_list.append(obj["label"])
                            # Add subclasses if available
                            if "possible_subclasses" in obj and obj["possible_subclasses"]:
                                target_list.extend(obj["possible_subclasses"])
                
                return {
                    "target_list": target_list,
                    "full_analysis": objects_info
                }
            else:
                raise ValueError("No JSON content found in response")
        except Exception as e:
            print(f"Error parsing LLM object identification: {e}")
            print(f"Raw response: {response}")
            return {"target_list": [], "full_analysis": {"identified_objects": []}}
            
    def analyze_targets(self, target_list, yolo_classes, image_base64=None):
        """Analyze targets and map them to YOLO classes.
        
        Implements the BaseProvider interface for target analysis.
        
        Args:
            target_list: List of target objects to detect
            yolo_classes: Dict[int, str] or List[str] of YOLO class names
            image_base64: Optional base64 encoded image
            
        Returns:
            dict: {
                "target_mappings": [{
                    "target": str,
                    "yolo_classes": List[str],
                    "attributes": List[str],
                    "subclasses": List[str],
                    "confidence_adjustments": Dict[str, float]
                }],
                "detection_strategy": str
            }
        """
        logger.info("Starting analyze_targets")
        logger.debug(f"Input target_list: {target_list}")
        logger.debug(f"Input yolo_classes: {yolo_classes}")
        
        # Ensure we have valid inputs
        if not target_list:
            logger.info("No target list provided, returning empty mapping")
            return {
                "target_mappings": [],
                "detection_strategy": "no_targets"
            }
            
        # Normalize target list to list format
        if isinstance(target_list, str):
            target_list = [target_list]
            logger.debug(f"Normalized target_list to: {target_list}")
            
        # Normalize YOLO classes to list format
        if isinstance(yolo_classes, dict):
            logger.debug("Converting yolo_classes from dict to list")
            classes_list = list(yolo_classes.values())
        elif isinstance(yolo_classes, (list, tuple)):
            classes_list = list(yolo_classes)
        else:
            classes_list = [str(yolo_classes)]
        logger.debug(f"Normalized classes_list: {classes_list}")
            
        # Remove any duplicates while preserving order
        seen = set()
        classes_list = [x for x in classes_list if not (x in seen or seen.add(x))]
        logger.debug(f"Deduplicated classes_list: {classes_list}")
        
        # Convert lists to JSON strings for prompt
        target_list_str = json.dumps(target_list)
        classes_list_str = json.dumps(classes_list)
        logger.debug(f"Target list JSON: {target_list_str}")
        logger.debug(f"Classes list JSON: {classes_list_str}")
        
        # Prepare prompt content
        content = []
        if image_base64:
            logger.debug("Adding image to prompt content")
            # Ensure image is properly encoded
            encoded_image = self._encode_image(image_base64)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded_image
                }
            })
            
        # Build prompt with explicit target and class lists
        prompt_text = ANALYZE_TARGETS_PROMPT.format(
            target_list=target_list_str,
            classes_list=classes_list_str
        )
        content.append({
            "type": "text",
            "text": prompt_text
        })
        logger.debug(f"Generated prompt:\n{prompt_text}")
        
        # Request JSON response from LLM
        prompt = [{"role": "user", "content": content}]
        logger.info("Sending request to LLM")
        response = self._make_llm_request(prompt, max_tokens=1000)
        
        try:
            # Log full response for debugging
            logger.debug(f"Raw LLM response:\n{json.dumps(response, indent=2)}")
            
            # Extract content from response
            content = response["content"][0]["text"].strip()
            print("\n=== RAW TEXT CONTENT ===")
            print(content)
            print("=== END TEXT CONTENT ===\n")
            
            # Find JSON content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                # Extract and parse JSON
                json_content = content[json_start:json_end]
                print("\n=== EXTRACTED JSON ===")
                print(json_content)
                print("=== END JSON ===\n")
                
                result = json.loads(json_content)
                logger.debug(f"Parsed result:\n{json.dumps(result, indent=2)}")
                
                # Validate response structure
                if not isinstance(result, dict):
                    logger.error("Response is not a dictionary")
                    raise ValueError("Response must be a dictionary")
                
                # Ensure target_mappings exists and is a list
                if "target_mappings" not in result or not isinstance(result["target_mappings"], list):
                    logger.warning("Missing or invalid target_mappings, creating defaults")
                    result["target_mappings"] = []
                    for target in target_list:
                        result["target_mappings"].append({
                            "target": target,
                            "yolo_classes": classes_list,
                            "attributes": [],
                            "subclasses": [],
                            "confidence_adjustments": {}
                        })
                
                # Ensure detection_strategy exists
                if "detection_strategy" not in result or not isinstance(result["detection_strategy"], str):
                    logger.warning("Missing or invalid detection_strategy, using default")
                    result["detection_strategy"] = "default_mapping"
                
                # Validate and normalize each mapping
                normalized_mappings = []
                for i, mapping in enumerate(result["target_mappings"]):
                    logger.debug(f"Processing mapping {i}: {mapping}")
                    
                    if not isinstance(mapping, dict):
                        logger.warning(f"Mapping {i} is not a dictionary, skipping")
                        continue
                        
                    # Create normalized mapping with all required fields
                    normalized_mapping = {
                        "target": mapping.get("target", "unknown"),
                        "yolo_classes": [cls for cls in mapping.get("yolo_classes", []) if cls in classes_list],
                        "attributes": mapping.get("attributes", []),
                        "subclasses": mapping.get("subclasses", []),
                        "confidence_adjustments": {
                            cls: float(adj) for cls, adj in mapping.get("confidence_adjustments", {}).items()
                            if cls in classes_list and isinstance(adj, (int, float))
                        }
                    }
                    logger.debug(f"Normalized mapping {i}: {normalized_mapping}")
                    
                    # Only include mappings with valid targets and at least one valid YOLO class
                    if normalized_mapping["target"] in target_list:
                        # If no valid YOLO classes were found, use all classes
                        if not normalized_mapping["yolo_classes"]:
                            logger.warning(f"No valid YOLO classes for target '{normalized_mapping['target']}', using all classes")
                            normalized_mapping["yolo_classes"] = classes_list
                        normalized_mappings.append(normalized_mapping)
                    else:
                        logger.warning(f"Invalid target '{normalized_mapping['target']}', skipping")
                
                # Update result with normalized mappings
                result["target_mappings"] = normalized_mappings
                logger.debug(f"Final normalized mappings:\n{json.dumps(normalized_mappings, indent=2)}")
                
                # If no valid mappings were created, create default mappings
                if not result["target_mappings"]:
                    logger.warning("No valid mappings created, using fallback mappings")
                    result["target_mappings"] = [{
                        "target": target,
                        "yolo_classes": classes_list,
                        "attributes": [],
                        "subclasses": [],
                        "confidence_adjustments": {}
                    } for target in target_list]
                    result["detection_strategy"] = "fallback_mapping"
                
                logger.info("Successfully processed LLM response")
                return result
            else:
                logger.error("No JSON content found in response")
                raise ValueError("No JSON content found in response")
                
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            logger.debug(f"Raw response that caused error:\n{json.dumps(response, indent=2)}")
            
            # Return safe default structure following the BaseProvider interface
            fallback = {
                "target_mappings": [{
                    "target": target,
                    "yolo_classes": classes_list,
                    "attributes": [],
                    "subclasses": [],
                    "confidence_adjustments": {}
                } for target in target_list],
                "detection_strategy": "error_fallback"
            }
            logger.info("Returning fallback response")
            return fallback
            
    def refine_detections(self, target_list, yolo_detections, original_image_base64, 
                         image_width, image_height, previous_refined=None, 
                         iteration=1, max_iterations=3):
        # Format inputs for prompt
        detections_json = json.dumps(yolo_detections)
        target_list_str = json.dumps(target_list) if isinstance(target_list, list) else json.dumps([target_list])
        
        # Find contained and overlapping boxes
        contained_boxes = []
        overlapping_boxes = []
        
        for i, det1 in enumerate(yolo_detections):
            for j, det2 in enumerate(yolo_detections):
                if i != j:
                    if self._is_contained_in(det1["box"], det2["box"]):
                        contained_boxes.append((i, j))
                    
                    iou = self._calculate_iou(det1["box"], det2["box"])
                    if iou > 0.3:
                        overlapping_boxes.append((i, j, iou))
        
        contained_boxes_json = json.dumps(contained_boxes)
        overlapping_boxes_json = json.dumps(overlapping_boxes)
        
        # Format previous results if available
        previous_json = "null"
        if previous_refined:
            previous_json = json.dumps(previous_refined)
            
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": self._encode_image(original_image_base64)
                }
            },
            {
                "type": "text",
                "text": prompt_manager.get_prompt(
                    "refine_detections",
                    REFINE_DETECTIONS_PROMPT.format(
                        target_list=target_list_str,
                        image_width=image_width,
                        image_height=image_height,
                        detections_json=detections_json,
                        contained_boxes_json=contained_boxes_json,
                        overlapping_boxes_json=overlapping_boxes_json,
                        iteration=iteration,
                        previous_json=previous_json
                    )
                )
            }
        ]
        
        prompt = [{"role": "user", "content": content}]
        
        response = self._make_llm_request(prompt, max_tokens=2000)
        try:
            content = response["content"][0]["text"]
            # Extract just the JSON part from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                # Ensure the response has the required structure
                if not isinstance(result, dict):
                    raise ValueError("Response must be a dictionary")
                    
                if "refined_detections" not in result:
                    result["refined_detections"] = []
                if "removed_detections" not in result:
                    result["removed_detections"] = []
                if "analysis" not in result:
                    result["analysis"] = {
                        "improvements": [],
                        "remaining_issues": ["Failed to process detections"],
                        "confidence_score": 0.0,
                        "continue_iteration": False
                    }
                    
                return result
            else:
                raise ValueError("No JSON content found in response")
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            return {
                "refined_detections": [],
                "removed_detections": [],
                "analysis": {
                    "improvements": [],
                    "remaining_issues": ["Failed to process detections"],
                    "confidence_score": 0.0,
                    "continue_iteration": False
                }
            }

    def _make_llm_request(self, messages, max_tokens=1000):
        """Make a request to the Anthropic API."""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": messages
                }
            )
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                raise Exception(f"Error: {response.status_code}\nResponse: {response.text}")
                
            result = response.json()
            
            # Print raw response text for debugging
            print("\n=== RAW LLM RESPONSE ===")
            print(json.dumps(result, indent=2))
            print("=== END RAW RESPONSE ===\n")
            
            return result
            
        except Exception as e:
            print(f"Error making API request: {str(e)}")
            raise

    def _encode_image(self, image_base64):
        """Ensure image is properly base64 encoded for the API."""
        # Remove any existing header
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
            
        # Remove any whitespace
        image_base64 = image_base64.strip()
        
        return image_base64
