from PIL import Image, ImageDraw, ImageFont

class BoxVisualizer:
    def __init__(self):
        self.colors = {}
        
    def draw_boxes(self, image_path, detections):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arialbd.ttf", 20)
        except:
            font = ImageFont.load_default()
            font_size = 20
            try:
                font = font.font_variant(size=font_size)
            except AttributeError:
                pass
        
        default_color = (255, 0, 0)
        
        for det in detections:
            # Handle both "class" and "label" fields for compatibility
            label = det.get("class", det.get("label", "unknown"))
            if label not in self.colors:
                self.colors[label] = default_color
        
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            # Handle both "class" and "label" fields for compatibility
            label = det.get("class", det.get("label", "unknown"))
            conf = det.get("confidence", 1.0)
            
            color = self.colors.get(label, default_color)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            text = f"{label} {conf:.2f}"
            
            if hasattr(draw, 'textsize'):
                text_w, text_h = draw.textsize(text, font=font)
            else:
                try:
                    text_w, text_h = font.getsize(text)
                except:
                    text_w, text_h = len(text) * 12, 20
            
            draw.rectangle([x1, y1, x1 + text_w, y1 + text_h], fill=color)
            draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
            
            if "reasoning" in det and det["reasoning"]:
                print(f"Reasoning for {label} at {det['box']}: {det['reasoning']}")
        
        return image
