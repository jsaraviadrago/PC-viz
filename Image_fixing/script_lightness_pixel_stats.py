import cv2
import numpy as np


def print_image_statistics(image_path):
    """
    Prints pixel statistics for the input image, including channel-wise color distribution.
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
        # Split the image into Blue, Green, and Red channels
        blue_channel, green_channel, red_channel = cv2.split(image)

        # Calculate total pixels
        total_pixels = image.shape[0] * image.shape[1]

        # Calculate color channel percentages and pixel counts
        blue_pixels = np.sum(blue_channel > 0)
        green_pixels = np.sum(green_channel > 0)
        red_pixels = np.sum(red_channel > 0)

        blue_percentage = (blue_pixels / total_pixels) * 100
        green_percentage = (green_pixels / total_pixels) * 100
        red_percentage = (red_pixels / total_pixels) * 100

        print("\n=== Color Channel Analysis ===")
        print(f"Total Pixels: {total_pixels}")
        print(f"Blue Pixels: {blue_pixels} ({blue_percentage:.2f}%)")
        print(f"Green Pixels: {green_pixels} ({green_percentage:.2f}%)")
        print(f"Red Pixels: {red_pixels} ({red_percentage:.2f}%)")

        # Calculate average intensity for each channel
        blue_avg = np.mean(blue_channel)
        green_avg = np.mean(green_channel)
        red_avg = np.mean(red_channel)

        print("\n=== Channel Average Intensities ===")
        print(f"Blue Channel Average: {blue_avg:.2f}")
        print(f"Green Channel Average: {green_avg:.2f}")
        print(f"Red Channel Average: {red_avg:.2f}")

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