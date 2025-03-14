import json
import requests
from .base import BaseProvider
from ..prompts.identify_objects import PROMPT as IDENTIFY_OBJECTS_PROMPT
from ..prompts.analyze_targets import PROMPT as ANALYZE_TARGETS_PROMPT
from ..prompts.refine_detections import PROMPT as REFINE_DETECTIONS_PROMPT
from ..prompts import prompt_manager

class LLM(BaseProvider):
    def __init__(self, api_key, model="gpt-4-vision-preview"):
        super().__init__(api_key, model)
        
    def identify_objects(self, image_base64):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
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
            content = response["choices"][0]["message"]["content"]
            objects_info = json.loads(content)
            
            target_list = [obj["label"] for obj in objects_info["identified_objects"]]
            for obj in objects_info["identified_objects"]:
                if "possible_subclasses" in obj and obj["possible_subclasses"]:
                    target_list.extend(obj["possible_subclasses"])
            
            return {
                "target_list": target_list,
                "full_analysis": objects_info
            }
        except Exception as e:
            print(f"Error parsing LLM object identification: {e}")
            print(f"Raw response: {response}")
            return {"target_list": [], "full_analysis": {"identified_objects": []}}
            
    def analyze_targets(self, target_list, yolo_classes, image_base64=None):
        classes_list = json.dumps(list(yolo_classes.values()))
        
        content = []
        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
            
        content.append({
            "type": "text",
            "text": prompt_manager.get_prompt(
                "analyze_targets",
                ANALYZE_TARGETS_PROMPT.format(
                    target_list=target_list,
                    classes_list=classes_list
                )
            )
        })
        
        prompt = [{"role": "user", "content": content}]
        
        response = self._make_llm_request(prompt, max_tokens=1000)
        try:
            content = response["choices"][0]["message"]["content"]
            mapping = json.loads(content)
            return mapping
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            return {"target_mappings": [], "detection_strategy": "failed"}
            
    def refine_detections(self, target_list, yolo_detections, original_image_base64, 
                         image_width, image_height, previous_refined=None, 
                         iteration=1, max_iterations=3):
        detections_json = json.dumps(yolo_detections)
        
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
        
        previous_json = "null"
        if previous_refined:
            previous_json = json.dumps(previous_refined)
            
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{original_image_base64}"
                }
            },
            {
                "type": "text",
                "text": prompt_manager.get_prompt(
                    "refine_detections",
                    REFINE_DETECTIONS_PROMPT.format(
                        target_list=target_list,
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
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            return {
                "refined_detections": [],
                "analysis": {
                    "improvements": [],
                    "remaining_issues": ["Failed to process detections"],
                    "continue_iteration": False
                }
            }

    def _make_llm_request(self, messages, max_tokens=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return {"choices": [{"message": {"content": "{}"}}]}
            
        return response.json()