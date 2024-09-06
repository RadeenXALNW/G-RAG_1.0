from PIL import Image 
import numpy as np


def is_image_valid(image_path, threshold=0.95, min_std=10):
    """
    Check if the image is valid (not mostly black, white, or low information).
    
    :param image_path: Path to the image file
    :param threshold: Threshold for considering an image as mostly black or white
    :param min_std: Minimum standard deviation to consider an image as containing enough information
    :return: Boolean indicating if the image is valid
    """
    with Image.open(image_path) as img:
        img_array = np.array(img.convert('L'))  # Convert to grayscale
        
        # Check if image is mostly black or white
        black_white_ratio = np.mean(img_array < 10) + np.mean(img_array > 245)
        if black_white_ratio > threshold:
            return False
        
        # Check if image has enough variation
        if np.std(img_array) < min_std:
            return False
        
    return True