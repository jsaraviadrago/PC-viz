import cv2
import numpy as np
import os


def load_colorization_model(model_dir):
    """
    Load the pre-trained colorization model from the given directory.
    """
    prototxt = os.path.join(model_dir, 'colorization_deploy_v2.prototxt')
    model = os.path.join(model_dir, 'colorization_release_v2.caffemodel')
    cluster_points = os.path.join(model_dir, '../resources/pts_in_hull.npy')

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
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Ensure L-channel is uint8 and apply CLAHE
    l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Ensure a_channel and b_channel match the type and size of l_channel
    a_channel = a_channel.astype(np.uint8)
    b_channel = b_channel.astype(np.uint8)

    # Match sizes explicitly (in case of any minor mismatches)
    a_channel = cv2.resize(a_channel, (l_channel.shape[1], l_channel.shape[0]))
    b_channel = cv2.resize(b_channel, (l_channel.shape[1], l_channel.shape[0]))

    # Merge the channels back together
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

    # Step 1: Normalize the L-channel for better dynamic range
    l_channel_normalized = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)

    # Step 2: Preprocess the L-channel
    l_channel_resized = cv2.resize(l_channel_normalized, (224, 224))  # Model expects 224x224 input
    l_channel_resized = np.clip(l_channel_resized - 50, 0, 255)

    # Step 3: Run the model
    input_blob = cv2.dnn.blobFromImage(l_channel_resized)
    model.setInput(input_blob)
    ab_channels = model.forward()[0, :, :, :].transpose((1, 2, 0))

    print("=== Debugging AB Channels ===")
    print(f"AB channel range: min={ab_channels.min()}, max={ab_channels.max()}")

    # Step 4: Amplify AB channels to boost color saturation
    ab_channels_resized = cv2.resize(ab_channels, (image.shape[1], image.shape[0]))
    ab_channels_resized *= 2.0  # Amplify saturation

    # Step 5: Combine L and AB channels
    colorized_lab = np.concatenate((l_channel[:, :, np.newaxis], ab_channels_resized), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)

    # Step 6: Post-process the output
    colorized_bgr = enhance_contrast(colorized_bgr)  # Enhance brightness and contrast

    # Optional: Sharpen the final output
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    colorized_bgr = cv2.filter2D(colorized_bgr, -1, kernel)

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
