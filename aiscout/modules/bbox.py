def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: List of [x1, y1, x2, y2] coordinates
        box2: List of [x1, y1, x2, y2] coordinates
        
    Returns:
        float: IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def is_contained_in(box1, box2):
    """Check if box1 is contained within box2.
    
    Args:
        box1: List of [x1, y1, x2, y2] coordinates
        box2: List of [x1, y1, x2, y2] coordinates
        
    Returns:
        bool: True if box1 is contained within box2
    """
    return (box1[0] >= box2[0] and box1[1] >= box2[1] and 
            box1[2] <= box2[2] and box1[3] <= box2[3])
