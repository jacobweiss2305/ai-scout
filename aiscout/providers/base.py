from abc import ABC, abstractmethod
from ..modules.bbox import calculate_iou, is_contained_in

class BaseProvider(ABC):
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        
    @abstractmethod
    def identify_objects(self, image_base64):
        """Identify objects in an image.
        
        Args:
            image_base64: Base64 encoded image data
            
        Returns:
            dict: Contains target_list and full_analysis
        """
        pass
        
    @abstractmethod
    def analyze_targets(self, target_list, yolo_classes, image_base64=None):
        """Map target objects to YOLO classes and analyze detection strategy.
        
        Args:
            target_list: List of target objects to detect
            yolo_classes: Dict of available YOLO classes
            image_base64: Optional base64 encoded image data
            
        Returns:
            dict: Contains target_mappings and detection_strategy
        """
        pass
        
    @abstractmethod
    def refine_detections(self, target_list, yolo_detections, original_image_base64, 
                         image_width, image_height, previous_refined=None, 
                         iteration=1, max_iterations=3):
        """Refine and improve object detections.
        
        Args:
            target_list: List of target objects
            yolo_detections: List of YOLO detection results
            original_image_base64: Base64 encoded original image
            image_width: Width of the image
            image_height: Height of the image
            previous_refined: Previous iteration results
            iteration: Current iteration number
            max_iterations: Maximum number of iterations
            
        Returns:
            dict: Contains refined_detections and analysis
        """
        pass
        
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        return calculate_iou(box1, box2)
        
    def _is_contained_in(self, box1, box2):
        """Check if box1 is contained within box2."""
        return is_contained_in(box1, box2)