import cv2
import numpy as np
import os
from skimage import measure


def calculate_image_sharpness(image):
    """
    Calculate image sharpness using Laplacian variance method

    Args:
        image (numpy.ndarray): Input image

    Returns:
        float: Sharpness score (higher is sharper)
    """
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def enhance_and_compare_sharpness(image_path):
    """
    Enhance image and compare sharpness of different methods

    Args:
        image_path (str): Path to input image

    Returns:
        dict: Enhanced images with their sharpness scores
    """
    # Read the original image
    original_img = cv2.imread(image_path)

    if original_img is None:
        raise ValueError(f"Unable to read image from {image_path}")

    # Get the directory and base filename
    output_dir = os.path.dirname(image_path)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Enhancement techniques
    techniques = {
        'original': original_img,
        'bicubic_upscaled': cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
        'lanczos_upscaled': cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4),
        'sharpened': cv2.filter2D(
            cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4),
            -1,
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        ),
        'denoised': cv2.fastNlMeansDenoisingColored(
            cv2.resize(original_img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4),
            None, 10, 10, 7, 21
        )
    }

    # Calculate sharpness for each technique
    sharpness_results = {}
    for name, img in techniques.items():
        sharpness = calculate_image_sharpness(img)
        sharpness_results[name] = {
            'image': img,
            'sharpness_score': sharpness
        }

        # Save enhanced images
        output_path = os.path.join(output_dir, f"{base_filename}_{name}_sharpness.png")
        cv2.imwrite(output_path, img)
        print(f"{name} - Sharpness Score: {sharpness:.2f}")

    return sharpness_results


# Process the image
image_path = '/home/juan-carlos/PycharmProjects/PC-viz/Image_fixing/Nico_Marinovich.png'

try:
    sharpness_comparison = enhance_and_compare_sharpness(image_path)

    # Find the sharpest method
    sharpest_method = max(sharpness_comparison, key=lambda k: sharpness_comparison[k]['sharpness_score'])
    print(f"\nSharpest method: {sharpest_method}")
except Exception as e:
    print(f"Error during sharpness comparison: {e}")