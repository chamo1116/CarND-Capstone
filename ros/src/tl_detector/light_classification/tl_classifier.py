import rospy
import os
import cv2
import numpy as np
import keras
import operator
from keras import applications
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from styx_msgs.msg import TrafficLight

window_param_sets = [
    dict(x_start_stop=[48, 752], y_start_stop=[0, 480], xy_window=(192, 192), xy_overlap=(0.75, 0.75)),
    dict(x_start_stop=[36, 762], y_start_stop=[256, 600], xy_window=(96, 96), xy_overlap=(0.5, 0.5))
]

num_bins = 16

# Code heavily inspired by:
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html#svm-opencv
def hog_for_img(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    # quantizing binvalues in (0...num_bins)
    bins = np.int32(num_bins * ang / (2 * np.pi))
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), num_bins) for b, m in zip(bin_cells, mag_cells)]
    # hist is a 64 bit vector
    hist = np.hstack(hists)
    return hist

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Code source: Vehicle Detection and Tracking lesson,
    30. Sliding Window Implementation
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(img, windows, classifier):
    on_windows = []
    for window in windows:
        test_img_crop = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        test_img = cv2.resize(test_img_crop, (32, 32))      
        features = np.float32(np.array([hog_for_img(test_img)]))
        result = int(classifier.predict(features)[1][0][0])
        if result == 1:
            on_windows.append(window)
    return on_windows

def index_for_max(values):
    index, value = max(enumerate(values), key=operator.itemgetter(1))
    return index

class TLClassifier(object):
    def __init__(self):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.svm = cv2.ml.SVM_load(os.path.join(this_dir, 'svm_data.dat'))
        self.cnn_bottom = applications.VGG16(include_top=False, weights=None)
        self.cnn_bottom.load_weights(os.path.join(this_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
        self.cnn_top = load_model(os.path.join(this_dir, 'bottleneck_fc_full_model.h5'))

    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            img (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        window_lists = [slide_window(img, **params) for params in window_param_sets]
        full_window_list = []
        for window_list in window_lists:
            full_window_list += window_list
        hot_windows = search_windows(img, full_window_list, self.svm)

        votes = [0, 0, 0]

        # https://github.com/fchollet/keras/issues/2397
        # handle weird multithreading async wonkiness
        graph = K.get_session().graph
        with graph.as_default():
            for window in hot_windows:
                test_img_crop = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
                test_img = cv2.resize(test_img_crop, (32, 32))
                x = image.img_to_array(test_img)
                x = preprocess_input(x)

                bottom_out = self.cnn_bottom.predict(np.array([x]))
                predictions = self.cnn_top.predict(bottom_out)[0]
                prediction = index_for_max(predictions)
                votes[prediction] += 1
        prediction = index_for_max(votes)
        rospy.logwarn('Prediction: {}'.format(prediction))

        return TrafficLight.UNKNOWN
