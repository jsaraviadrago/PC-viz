import cv2
import numpy as np
import os


def adaptive_sharpening(image, sigma=1.0, amount=1.5):
    """
    Perform adaptive unsharp masking for more natural sharpening
    """
    # Convert to float
    image_float = image.astype(np.float32) / 255.0

    # Gaussian blur
    blurred = cv2.GaussianBlur(image_float, (0, 0), sigma)

    # Calculate unsharp mask
    unsharp_mask = image_float - blurred

    # Sharpen image
    sharpened = image_float + amount * unsharp_mask

    # Clip and convert back to uint8
    sharpened = np.clip(sharpened * 255.0, 0, 255).astype(np.uint8)

    return sharpened


def measure_sharpness(image):
    """
    Measure sharpness using the variance of the Laplacian
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def measure_snr(image):
    """
    Measure signal-to-noise ratio (SNR)
    """
    mean_signal = np.mean(image)
    std_noise = np.std(image)
    snr = mean_signal / std_noise if std_noise != 0 else float('inf')
    return snr


def measure_contrast(image):
    """
    Measure contrast using Michelson contrast
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity + 1e-5)
    return contrast


def edge_density(image):
    """
    Measure edge density using Canny edge detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def enhance_image(image_path):
    """
    Enhanced image processing with multiple techniques and diagnostics
    """
    # Read the original image
    original_img = cv2.imread(image_path)

    if original_img is None:
        raise ValueError(f"Unable to read image from {image_path}")

    # Get the directory of the original image
    output_dir = os.path.dirname(image_path)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Lanczos Upscaling with improved parameters
    upscaled = cv2.resize(original_img, None,
                          fx=2, fy=2,
                          interpolation=cv2.INTER_LANCZOS4)

    # 2. Adaptive Sharpening
    sharpened_mild = adaptive_sharpening(upscaled, sigma=1.0, amount=0.7)
    sharpened_moderate = adaptive_sharpening(upscaled, sigma=1.0, amount=1.5)
    sharpened_strong = adaptive_sharpening(upscaled, sigma=1.0, amount=2.5)

    # 3. Noise Reduction with adaptive parameters
    denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)

    # 4. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))  # Convert tuple to list for modification
    lab_planes[0] = clahe.apply(lab_planes[0])  # Apply CLAHE to the L channel
    lab = cv2.merge(lab_planes)  # Merge the planes back together
    contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Collect and save results
    enhanced_images = {
        'original': original_img,
        'upscaled': upscaled,
        'denoised': denoised,
        'contrast_enhanced': contrast_enhanced,
        'sharpened_mild': sharpened_mild,
        'sharpened_moderate': sharpened_moderate,
        'sharpened_strong': sharpened_strong
    }

    # Run diagnostics and save images
    diagnostics = {}
    for name, img in enhanced_images.items():
        output_path = os.path.join(output_dir, f"{base_filename}_{name}_enhanced.png")
        cv2.imwrite(output_path, img)
        print(f"Saved {name} enhanced image")

        # Diagnostics
        diagnostics[name] = {
            'sharpness': measure_sharpness(img),
            'snr': measure_snr(img),
            'contrast': measure_contrast(img),
            'edge_density': edge_density(img)
        }

    # Print diagnostics
    print("\nImage Diagnostics:")
    for name, metrics in diagnostics.items():
        print(f"{name}: {metrics}")

    # Return enhanced images and diagnostics
    return enhanced_images, diagnostics


# Process the image
image_path = '/home/juan-carlos/PycharmProjects/PC-viz/Image_fixing/Nico_Marinovich.png'

try:
    enhanced_images, diagnostics = enhance_image(image_path)
    print("\nImage enhancement and diagnostics completed successfully!")
except Exception as e:
    print(f"Error during image enhancement: {e}")
    import traceback
    traceback.print_exc()
