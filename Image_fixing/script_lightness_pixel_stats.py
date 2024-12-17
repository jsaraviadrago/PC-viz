import cv2
import numpy as np


def print_image_statistics(image_path):
    """
    Prints pixel statistics for the input image and its LAB Lightness channel.
    """
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at: {image_path}")

    print("=== Input Image Statistics ===")
    print(f"Image shape: {image.shape}")
    print(f"Pixel range (BGR): min={image.min()}, max={image.max()}")

    # Check if the image is grayscale or color
    if len(image.shape) == 2 or image.shape[2] == 1:
        print("The image is grayscale.")
    else:
        print("The image is color (BGR).")

    # Convert to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = image_lab[:, :, 0]  # Extract L (lightness) channel

    print("\n=== LAB Channel Statistics ===")
    print(f"L-channel shape: {l_channel.shape}")
    print(f"L-channel range: min={l_channel.min()}, max={l_channel.max()}")

    # Optional: Display the L-channel for visual inspection
    cv2.imshow("L-channel (Lightness)", l_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Replace this path with the path to your input image
image_path = '/home/juan-carlos/PycharmProjects/PC-viz/Image_fixing/Nico_Marinovich_contrast_enhanced_enhanced.png'

# Run the script
try:
    print_image_statistics(image_path)
except Exception as e:
    print(f"Error: {e}")
