import base64
from io import BytesIO
from PIL import Image

def encode_image(image_path):
    # Open image with PIL to handle format conversion
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (handles RGBA PNG images)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                bg.paste(img, mask=img.split()[3])
            else:
                bg.paste(img)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Save as JPEG to BytesIO
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))
