import cv2
import numpy as np
import os


def load_colorization_model(model_dir):
    """
    Load the pre-trained colorization model from the given directory.
    """
    prototxt = os.path.join(model_dir,
                            '/home/juan-carlos/PycharmProjects/PC-viz/colorization/models/colorization_deploy_v2.prototxt')
    model = os.path.join(model_dir,
                         '/home/juan-carlos/PycharmProjects/PC-viz/colorization/models/colorization_release_v2.caffemodel')
    cluster_points = os.path.join(model_dir,
                                  '/home/juan-carlos/PycharmProjects/PC-viz/colorization/resources/pts_in_hull.npy')

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

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Merge enhanced L-channel back with A and B channels
    enhanced_lab = cv2.merge((l_channel
