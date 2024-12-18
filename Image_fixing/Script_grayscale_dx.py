import cv2
import numpy as np


def analyze_image_color_nature(image_path):
    """
    Analyzes the color nature of an image and checks if it's effectively grayscale.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at: {image_path}")

    # Split the image into Blue, Green, and Red channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Calculate channel differences
    diff_blue_green = np.abs(blue_channel.astype(int) - green_channel.astype(int))
    diff_green_red = np.abs(green_channel.astype(int) - red_channel.astype(int))
    diff_blue_red = np.abs(blue_channel.astype(int) - red_channel.astype(int))

    # Calculate maximum difference across all pixels
    max_diff_blue_green = np.max(diff_blue_green)
    max_diff_green_red = np.max(diff_green_red)
    max_diff_blue_red = np.max(diff_blue_red)

    # Threshold for considering an image grayscale (adjust if needed)
    diff_threshold = 1  # Very small tolerance for differences

    print("=== Grayscale Analysis ===")
    print(f"Maximum B-G difference: {max_diff_blue_green}")
    print(f"Maximum G-R difference: {max_diff_green_red}")
    print(f"Maximum B-R difference: {max_diff_blue_red}")

    # Determine if the image is effectively grayscale
    is_grayscale = (max_diff_blue_green <= diff_threshold and
                    max_diff_green_red <= diff_threshold and
                    max_diff_blue_red <= diff_threshold)

    print(f"\nIs image effectively grayscale? {is_grayscale}")

    # Convert to grayscale if highly uniform
    if is_grayscale:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save grayscale image
        output_path = image_path.replace('.png', '_grayscale.png')
        cv2.imwrite(output_path, grayscale_image)
        print(f"\nConverted and saved grayscale image to: {output_path}")

        # Display some grayscale statistics
        print("\n=== Grayscale Image Statistics ===")
        print(f"Grayscale image shape: {grayscale_image.shape}")
        print(f"Pixel range: min={grayscale_image.min()}, max={grayscale_image.max()}")
    else:
        print("\nImage is not uniform enough to be considered purely grayscale.")


# Replace this path with the path to your input image
image_path = '/home/juan-carlos/PycharmProjects/PC-viz/Image_fixing/Nico_Marinovich_contrast_enhanced_enhanced.png'

# Run the script
try:
    analyze_image_color_nature(image_path)
except Exception as e:
    print(f"Error: {e}")