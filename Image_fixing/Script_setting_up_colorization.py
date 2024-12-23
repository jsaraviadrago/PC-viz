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


def colorize_image(image_path, model):
    """
    Colorize a grayscale image using the pre-trained model.
    """
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = image_lab[:, :, 0]  # Extract the L (lightness) channel

    # Preprocess the L channel for the model
    l_channel_resized = cv2.resize(l_channel, (224, 224))  # Model expects 224x224 input
    l_channel_resized = l_channel_resized - 50  # Subtract mean
    input_blob = cv2.dnn.blobFromImage(l_channel_resized)

    # Run the model
    model.setInput(input_blob)
    ab_channels = model.forward()[0, :, :, :].transpose((1, 2, 0))  # Output ab channels

    # Resize ab channels to match the original image size
    ab_channels_resized = cv2.resize(ab_channels, (image.shape[1], image.shape[0]))

    # Combine L and ab channels
    colorized_lab = np.concatenate((l_channel[:, :, np.newaxis], ab_channels_resized), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR

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
    model_dir = './models/'  # Update to the models directory path
    input_image = '/home/juan-carlos/PycharmProjects/PC-viz/Image_fixing/Nico_Marinovich_contrast_enhanced_enhanced.png'  # Replace with the path to your grayscale image
    output_image = './colorized_image.png'  # Output path for the colorized image

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
