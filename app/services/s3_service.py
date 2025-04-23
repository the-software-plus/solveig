# app/services/s3_service.py
import requests
from PIL import Image
import io

def download_image_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error downloading image from URL: {str(e)}")
        return None

def load_image_from_path(path):
    try:
        image = Image.open(path)
        return image
    except Exception as e:
        print(f"Error loading image from path: {str(e)}")
        return None