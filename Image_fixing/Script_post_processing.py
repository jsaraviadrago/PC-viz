import cv2
import numpy as np
import os


def load_colorization_model(model_dir):
    """
    Load the pre-trained colorization model from the given directory.
    """
    prototxt = os.path.join(model_dir, '/home/juan-carlos/PycharmProjects/PC-viz/colorization/models/colorization_deploy_v2.prototxt')
    model = os.path.join(model_dir, '/home/juan-carlos/PycharmProjects/PC-viz/colorization/models/colorization_release_v2.caffemodel')
    cluster_points = os.path.join(model_dir, '/home/juan-carlos/PycharmProjects/PC-viz/colorization/resources/pts_in_hull.npy')

    # Verify the existence of model files
    for file_path in [prototxt, model, cluster_points]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

    # Load the Caffe model
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Load cluster centers for AB color space and set up the model
    pts = np.load(cluster_points).transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

    return net


def enhance_contrast(image):
    """
    Enhance the brightness and contrast of the output image using CLAHE.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Ensure L-channel is of type uint8
    l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Merge enhanced L-channel back with A and B channels
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr



def colorize_image(image_path, model):
    """
    Colorize a grayscale image using the pre-trained model.
    """
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert to LAB color space and extract the L-channel
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = image_lab[:, :, 0]

    print("=== Debugging Input Image ===")
    print(f"L-channel range: min={l_channel.min()}, max={l_channel.max()}")

    # Preprocess the L-channel: resize and subtract mean
    l_channel_resized = cv2.resize(l_channel, (224, 224))  # Model expects 224x224 input
    l_channel_resized = np.clip(l_channel_resized - 50, 0, 255)  # Ensure valid range

    # Prepare the input blob
    input_blob = cv2.dnn.blobFromImage(l_channel_resized)

    # Run the model
    model.setInput(input_blob)
    ab_channels = model.forward()[0, :, :, :].transpose((1, 2, 0))  # Extract AB channels

    print("=== Debugging AB Channels ===")
    print(f"AB channel range: min={ab_channels.min()}, max={ab_channels.max()}")

    # Check AB channel validity
    if ab_channels.min() == 0 and ab_channels.max() == 0:
        print("Warning: AB channels contain only zeros. Model output may be invalid.")

    # Resize AB channels to match the original image size
    ab_channels_resized = cv2.resize(ab_channels, (image.shape[1], image.shape[0]))

    # Combine L and AB channels
    colorized_lab = np.concatenate((l_channel[:, :, np.newaxis], ab_channels_resized), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)

    # Enhance the output using CLAHE
    colorized_bgr = enhance_contrast(colorized_bgr)

    # Clip values to ensure valid range
    colorized_bgr = np.clip(colorized_bgr, 0, 255).astype('uint8')
    return colorized_bgr


def save_image(output_path, image):
    """
    Save the processed image to disk.
    """
    cv2.imwrite(output_path, image)
    print(f"Colorized image saved to: {output_path}")


# Main Pipeline
if __name__ == "__main__":
    model_dir = '/home/juan-carlos/PycharmProjects/PC-viz/colorization/models/'  # Path to model files
    input_image = '/home/juan-carlos/PycharmProjects/PC-viz/Image_fixing/Nico_Marinovich_contrast_enhanced_enhanced.png'
    output_image = './colorized_image_fixed.png'  # Output path for the colorized image

    try:
        # Load the model
        model = load_colorization_model(model_dir)

        # Colorize the image
        colorized_image = colorize_image(input_image, model)

        # Save the result
        save_image(output_image, colorized_image)

        print("\nImage colorization completed successfully!")

    except Exception as e:
        print(f"Error during colorization: {e}")
        import traceback

        traceback.print_exc()
