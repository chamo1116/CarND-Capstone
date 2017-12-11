import os
import cv2
import numpy as np

root_dir = '/capstone/training'

tl_base_dir = os.path.join(root_dir, 'traffic-lights-preprocessed')
non_tl_base_dir = os.path.join(root_dir, 'non-traffic-lights-preprocessed')

SPLIT_TRAIN = 'train'
SPLIT_TEST = 'test'

LABEL_TL = 1
LABEL_NON_TL = 0

bad_filenames = ['.DS_Store']

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

def shuffled_fl_for_f_and_l(features, labels):
    assert len(features) == len(labels)
    p = np.random.permutation(len(features))
    return (features[p], labels[p])

def features_and_labels_for_dir_and_label(directory, label):
    filenames = [f for f in os.listdir(directory) if f not in bad_filenames]
    imgs = [cv2.imread(os.path.join(directory, filename)) for filename in filenames]
    features = np.float32(np.array([hog_for_img(img) for img in imgs]))
    labels = np.array([label] * len(features))
    return (features, labels)

def data_for_split(split_name):
    tl_input_dir = os.path.join(tl_base_dir, split_name)
    non_tl_input_dir = os.path.join(non_tl_base_dir, split_name)
    tl_features, tl_labels = features_and_labels_for_dir_and_label(tl_input_dir, LABEL_TL)
    non_tl_features, non_tl_labels = features_and_labels_for_dir_and_label(non_tl_input_dir, LABEL_NON_TL)
    all_features = np.concatenate((tl_features, non_tl_features))
    all_labels = np.concatenate((tl_labels, non_tl_labels))
    shuffled_features, shuffled_labels = shuffled_fl_for_f_and_l(all_features, all_labels)
    return (shuffled_features, shuffled_labels)

def main():
    (train_features, train_labels) = data_for_split(SPLIT_TRAIN)
    (test_features, test_labels) = data_for_split(SPLIT_TEST)

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)

    svm.save('svm_data.dat')

    result = svm.predict(test_features)[1]
    result = np.append([], result) # TODO do this a better way
    mask = result == test_labels
    correct = np.count_nonzero(mask)
    print(correct * 100. / result.size)

if __name__ == '__main__':
    main()
