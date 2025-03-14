import time
import logging
from PIL import Image
from ultralytics import YOLO

from .modules.image_utils import encode_image
from .modules.visualization import BoxVisualizer
from .modules.color_analysis import ColorAnalyzer
from .modules.debug_output import (
    create_progress,
    print_detection_start,
    print_yolo_results,
    print_iteration_results,
    print_final_results
)
from typing import Optional, List, Callable, Literal

# Configure logging
logger = logging.getLogger(__name__)

class Scout:
    """A lightning fast and insanely accurate agentic object detection system.
    
    The Scout class follows a dependency injection pattern, taking an LLM instance directly
    for more flexible and testable code. Each provider (Anthropic and OpenAI) implements 
    the same LLM class name and BaseProvider interface.
    
    Example:
        ```python
        from aiscout import Scout
        from aiscout.providers.anthropic import LLM

        # Initialize LLM
        llm = LLM(api_key="your_api_key", model="claude-3-7-sonnet-20250219")

        # Initialize detector
        scout = Scout(llm=llm)

        # Run detection
        detections = scout.detect("path/to/image.jpg")
        ```
    
    Args:
        llm: LLM provider instance (Anthropic or OpenAI)
        yolo_model: YOLO model to use (default: "yolov8x.pt")
        confidence_threshold: Confidence threshold for YOLO detection (default: 0.25)
        debug_mode: Enable rich debug output (default: False)
    """
    def __init__(self, llm, yolo_model="yolov8x.pt", confidence_threshold=0.25, debug_mode=False):
        """Initialize Scout with an LLM provider.
        
        Args:
            llm: LLM provider instance (Anthropic or OpenAI)
            yolo_model: YOLO model to use (default: "yolov8x.pt")
            confidence_threshold: Confidence threshold for YOLO detection (default: 0.25)
            debug_mode: Enable rich debug output (default: False)
            
        Example:
            ```python
            from aiscout import Scout
            from aiscout.providers.anthropic import LLM

            # Initialize LLM
            llm = LLM(api_key="your_api_key", model="claude-3-7-sonnet-20250219")

            # Initialize detector
            scout = Scout(llm=llm)

            # Run detection
            detections = scout.detect("path/to/image.jpg")
            ```
        """
        # Initialize YOLO
        self.load_model(yolo_model)
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        
        # Initialize components
        self.box_visualizer = BoxVisualizer()
        self.color_analyzer = ColorAnalyzer()
        self.llm = llm
        
        logger.info(f"Scout initialized with model {yolo_model} and confidence threshold {confidence_threshold}")
        
    def load_model(self, yolo_model):
        self.model = YOLO(yolo_model)
        logger.debug(f"Loaded YOLO model: {yolo_model}")
        
    def run_yolo_detection(self, image_path, conf=None):
        conf = conf or self.confidence_threshold
        results = self.model(image_path, conf=conf)
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = result.names[cls]
                
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "class": cls_name,
                    "class_id": cls
                })
        
        logger.debug(f"YOLO detection results: {detections}")
        return detections
    
    def get_yolo_target_list(self, detections, max_targets=10):
        labels = {}
        for det in detections:
            cls = det["class"]
            conf = det["confidence"]
            if cls not in labels or conf > labels[cls]:
                labels[cls] = conf
        
        sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
        return [label for label, _ in sorted_labels[:max_targets]]
    
    def detect(self, 
              image_path: str, 
              target_list: Optional[List[str]] = None, 
              conf: Optional[float] = None,
              min_iterations: int = 1,
              max_iterations: int = 3,
              confidence_threshold: float = 0.9,
              auto_target_source: Literal["llm", "yolo"] = "llm",
              progress_callback: Optional[Callable] = None):
        """Run object detection with iterative refinement.
        
        This is the main method for running object detection. It performs the following steps:
        1. Run initial YOLO detection
        2. Identify target objects (from provided list, LLM, or YOLO)
        3. Analyze and map targets to YOLO classes
        4. Iteratively refine detections using the LLM
        5. Draw final detections on the image
        
        Args:
            image_path: Path to the image file
            target_list: Optional list of target objects to detect
            conf: Optional confidence threshold for YOLO detection
            min_iterations: Minimum number of refinement iterations
            max_iterations: Maximum number of refinement iterations
            confidence_threshold: Score threshold to consider detection converged
            auto_target_source: Source for automatic target identification if no target_list ("llm" or "yolo")
            progress_callback: Optional callback for progress updates
            
        Returns:
            dict: Detection results including:
                - annotated_image: PIL Image with drawn detections
                - detections: List of refined detections
                - removed_detections: List of detections that were removed
                - yolo_raw_detections: Original YOLO detections before refinement
                - target_list: List of target objects to detect
                - target_mapping: Mapping between target objects and YOLO classes
                - scene_description: Optional scene description from LLM
                - object_identification: Full object identification results from LLM
                - iteration_history: History of refinement iterations
                - processing_time: Total processing time in seconds
                - target_source: Source of target list ("llm", "yolo", or "user")
        """
        start_time = time.time()
        logger.info("Starting detection process")
        
        # Print debug info if enabled
        if self.debug_mode:
            config = {
                "image_path": image_path,
                "target_list": target_list,
                "confidence_threshold": confidence_threshold,
                "min_iterations": min_iterations,
                "max_iterations": max_iterations,
                "auto_target_source": auto_target_source
            }
            print_detection_start(image_path, config)
            progress = create_progress()
            task = progress.add_task("Running detection...", total=max_iterations)
        
        if progress_callback:
            progress_callback({
                "status": "Starting detection process...",
                "iteration": 0,
                "max_iterations": max_iterations
            })
            
        image = Image.open(image_path)
        width, height = image.size
        image_base64 = encode_image(image_path)
        
        class_names = self.model.names

        # Run YOLO detection early so we can use it for either target source
        if progress_callback:
            progress_callback({
                "status": "Running YOLO detection...",
                "iteration": 0,
                "max_iterations": max_iterations
            })
        yolo_detections = self.run_yolo_detection(image_path, conf=conf)
        logger.info(f"YOLO detected {len(yolo_detections)} objects")
        
        # Add color analysis
        yolo_detections = self.color_analyzer.add_color_analysis(image_path, yolo_detections)
        
        # If no target list is provided, determine targets based on specified source
        if target_list is None:
            logger.info(f"No target list provided. Using {auto_target_source} to identify targets")
            
            if auto_target_source.lower() == "yolo":
                # Use YOLO's detected classes as targets
                target_list = self.get_yolo_target_list(yolo_detections)
                logger.info(f"YOLO identified targets: {target_list}")
                scene_description = None
                object_identification = None
            else:  # llm
                if progress_callback:
                    progress_callback({
                        "status": "Identifying targets using LLM...",
                        "iteration": 0,
                        "max_iterations": max_iterations
                    })
                
                object_identification = self.llm.identify_objects(image_base64)
                target_list = object_identification["target_list"]
                logger.info(f"LLM identified targets: {target_list}")
                scene_description = object_identification["full_analysis"].get("detected_scene_type", None)
        else:
            scene_description = None
            object_identification = None
            
        if self.debug_mode:
            print_yolo_results(
                len(yolo_detections),
                target_list,
                auto_target_source if target_list is None else "user"
            )
        
        # Analyze targets with LLM
        if progress_callback:
            progress_callback({
                "status": "Analyzing targets with LLM...",
                "iteration": 0,
                "max_iterations": max_iterations
            })
        target_mapping = self.llm.analyze_targets(target_list, class_names, image_base64)
        
        # Iterative refinement process
        iteration = 1
        previous_result = None
        final_result = None
        iteration_history = []
        
        while iteration <= max_iterations:
            logger.info(f"Starting iteration {iteration}/{max_iterations}")
            
            # Get refinements from LLM
            refined = self.llm.refine_detections(
                target_list,
                yolo_detections,
                image_base64,
                width,
                height,
                previous_refined=previous_result,
                iteration=iteration,
                max_iterations=max_iterations
            )
            
            previous_result = refined
            final_result = refined
            
            # Update detections
            yolo_detections = refined["refined_detections"]
            
            # Get iteration summary
            summary = refined.get("analysis", {})
            confidence_score = summary.get("confidence_score", 0.0)
            
            if progress_callback:
                progress_callback({
                    "status": "Iteration complete",
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "improvements": summary.get('improvements', []),
                    "remaining_issues": summary.get('remaining_issues', []),
                    "confidence_score": confidence_score
                })
            
            logger.info(f"Iteration {iteration} complete:")
            logger.info(f"  - Improvements: {summary.get('improvements', 'N/A')}")
            logger.info(f"  - Remaining issues: {summary.get('remaining_issues', 'N/A')}")
            logger.info(f"  - Confidence score: {confidence_score:.2f}")
            
            if self.debug_mode:
                print_iteration_results(
                    iteration,
                    max_iterations,
                    len(refined["refined_detections"]),
                    len(refined.get("removed_detections", [])),
                    confidence_score,
                    summary.get('improvements', []),
                    summary.get('remaining_issues', [])
                )
                progress.update(task, completed=iteration)
            
            iteration_history.append({
                "iteration": iteration,
                "detections_count": len(refined["refined_detections"]),
                "removed_count": len(refined.get("removed_detections", [])),
                "confidence_score": confidence_score,
                "summary": summary
            })
            
            # Early stopping if converged and minimum iterations met
            if iteration >= min_iterations and confidence_score >= confidence_threshold:
                logger.info(f"Confidence threshold reached ({confidence_score:.2f} >= {confidence_threshold})")
                break
                
            iteration += 1
        
        # Draw final detections
        if progress_callback:
            progress_callback({
                "status": "Drawing detections...",
                "iteration": iteration,
                "max_iterations": max_iterations
            })
        annotated_image = self.box_visualizer.draw_boxes(image_path, yolo_detections)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Detection completed in {processing_time:.2f} seconds")
        
        # Prepare results
        results = {
            "annotated_image": annotated_image,
            "detections": final_result["refined_detections"],
            "removed_detections": final_result.get("removed_detections", []),
            "yolo_raw_detections": yolo_detections,
            "target_list": target_list,
            "target_mapping": target_mapping,
            "scene_description": scene_description,
            "object_identification": object_identification,
            "iteration_history": iteration_history,
            "processing_time": processing_time,
            "target_source": auto_target_source if target_list is None else "user"
        }
        
        # Print final results if debug mode is enabled
        if self.debug_mode:
            progress.update(task, completed=max_iterations)
            progress.stop()
            print_final_results(results)
            
        return results
    
    def explain_refinements(self, result):
        """Generate a human-readable explanation of the refinement process"""
        if not result or "iteration_history" not in result:
            return "No refinement history available"
            
        explanation = []
        explanation.append(f"Processing completed in {result['processing_time']:.2f} seconds")
        
        if result.get("scene_description"):
            explanation.append(f"\nScene Description: {result['scene_description']}")
            
        explanation.append(f"\nTarget Objects: {', '.join(result['target_list'])}")
        
        explanation.append("\nRefinement Process:")
        for history in result["iteration_history"]:
            iteration = history["iteration"]
            detections = history["detections_count"]
            removed = history["removed_count"]
            score = history["confidence_score"]
            summary = history["summary"]
            
            explanation.append(f"\nIteration {iteration}:")
            explanation.append(f"  - Detections: {detections}")
            explanation.append(f"  - Removed: {removed}")
            explanation.append(f"  - Confidence: {score:.2f}")
            
            if "improvements" in summary:
                explanation.append("  - Improvements:")
                for imp in summary["improvements"]:
                    explanation.append(f"    * {imp}")
                    
            if "remaining_issues" in summary:
                explanation.append("  - Remaining Issues:")
                for issue in summary["remaining_issues"]:
                    explanation.append(f"    * {issue}")
                    
        return "\n".join(explanation)