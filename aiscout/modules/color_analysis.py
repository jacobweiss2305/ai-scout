import cv2

class ColorAnalyzer:
    def add_color_analysis(self, image_path, detections):
        """Add HSV color analysis to detections"""
        image = cv2.imread(image_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            roi = image_hsv[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            mean_hsv = cv2.mean(roi)[:3]
            h, s, v = mean_hsv
            
            h_hist = cv2.calcHist([roi], [0], None, [30], [0, 180])
            h_hist = h_hist.flatten() / h_hist.sum()
            
            det["color_data"] = {
                "mean_hsv": mean_hsv,
                "mean_h": h,
                "mean_s": s,
                "mean_v": v,
                "h_histogram": h_hist.tolist(),
                "dominant_hue_ranges": self._get_dominant_hue_ranges(h_hist)
            }
        
        return detections

    def _get_dominant_hue_ranges(self, h_hist, threshold=0.1):
        """Extract dominant hue ranges from histogram"""
        ranges = []
        max_val = max(h_hist)
        threshold_val = max_val * threshold
        
        for i, val in enumerate(h_hist):
            if val > threshold_val:
                hue_min = (i * 6)
                hue_max = hue_min + 6
                ranges.append({
                    "range": [hue_min, hue_max],
                    "strength": float(val / max_val)
                })
        
        return ranges
