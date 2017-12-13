import os
import cv2
import math
import numpy as np

root_dir = '/capstone/training'

tl_input_dir = os.path.join(root_dir, 'traffic-lights-raw')
non_tl_input_dir = os.path.join(root_dir, 'non-traffic-lights-raw/GTI')

tl_output_dirname = 'traffic-lights-preprocessed'
non_tl_output_dirname = 'non-traffic-lights-preprocessed'

train_dirname = 'train'
test_dirname = 'test'

num_total_samples = 697
num_train_samples = 500

bad_filenames = ['.DS_Store']

def create_output_dirs():
    for output_dirname in [tl_output_dirname, non_tl_output_dirname]:
        for split_dirname in [train_dirname, test_dirname]:
            output_dir = os.path.join(root_dir, output_dirname, split_dirname)
            if os.path.exists(output_dir):
                print('Directory {} exists'.format(output_dir))
            else:
                print('Creating directory {} ...'.format(output_dir))
                os.makedirs(output_dir)

def split_dirname_for_idx(idx):
    if idx < num_train_samples - 1:
        return train_dirname
    return test_dirname

def cropped_img_for_img(img):
    height, width, nb_channels = img.shape
    if height < width:
        excess = width - height
        crop_left_idx = int(math.floor(excess / 2.))
        crop_right_idx = width - int(math.ceil(excess / 2.))
        return img[:, crop_left_idx:crop_right_idx]
    if height > width:
        excess = height - width
        crop_top_idx = int(math.floor(excess / 2.))
        crop_bottom_idx = height - int(math.ceil(excess / 2.))
        return img[crop_top_idx:crop_bottom_idx, :]
    return img[:, :]

def preprocessed_img_for_img(img):
    cropped_img = cropped_img_for_img(img)
    preprocessed_img = cv2.resize(cropped_img, (32, 32), interpolation = cv2.INTER_CUBIC)
    return preprocessed_img

def shuffled_truncated_list_for_list(src_list, cap):
    # create shallow clone of list
    shuffled_list = src_list[:]
    np.random.shuffle(shuffled_list)
    return shuffled_list[:cap]

def preprocess_image_set(input_dir, output_dirname):
    raw_filenames = [f for f in os.listdir(input_dir) if f not in bad_filenames]
    shuffled_filenames = shuffled_truncated_list_for_list(raw_filenames, num_total_samples)
    for idx in range(len(shuffled_filenames)):
        input_filename = shuffled_filenames[idx]
        input_filepath = os.path.join(input_dir, input_filename)
        img = cv2.imread(input_filepath)
        preprocessed_img = preprocessed_img_for_img(img)
        split_dirname = split_dirname_for_idx(idx)
        output_filename = '{}.png'.format(idx)
        output_filepath_parts = (root_dir, output_dirname, split_dirname, output_filename);
        output_filepath = os.path.join(*output_filepath_parts)
        cv2.imwrite(output_filepath, preprocessed_img)

def main():
    create_output_dirs()
    preprocess_image_set(tl_input_dir, tl_output_dirname)
    preprocess_image_set(non_tl_input_dir, non_tl_output_dirname)

if __name__ == '__main__':
    main()
