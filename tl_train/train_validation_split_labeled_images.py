import os
import random
import shutil

root_dir = '/capstone/training'
input_dir = os.path.join(root_dir, 'traffic-lights-preprocessed-labeled')
output_dirname = 'labeled-split-images'
output_dir = os.path.join(root_dir, output_dirname)

SPLIT_TRAIN = 'train'
SPLIT_VAL = 'validation'

LABEL_RED = '0'
LABEL_YELLOW = '1'
LABEL_GREEN = '2'

splits = [SPLIT_TRAIN, SPLIT_VAL]
label_names = [l for l in [LABEL_RED, LABEL_YELLOW, LABEL_GREEN]]
bad_filenames = ['.DS_Store']

num_train_samples = 500

def create_output_dirs():
    for split in splits:
        for label in label_names:
            output_dir = os.path.join(root_dir, output_dirname, split, label)
            if os.path.exists(output_dir):
                print('Directory {} exists'.format(output_dir))
            else:
                print('Creating directory {} ...'.format(output_dir))
                os.makedirs(output_dir)

def shuffled_fl_for_f_and_l(filenames, labels):
    assert len(filenames) == len(labels)
    p = np.random.permutation(len(filenames))
    return (filenames[p], labels[p])

def fl_for_directory_and_label(base_directory, label):
    directory = os.path.join(base_directory, label)
    filenames = [f for f in os.listdir(directory) if f not in bad_filenames]
    labels = [label] * len(filenames)
    return (filenames, labels)

def data_for_directory(directory):
    r_filenames, r_labels = fl_for_directory_and_label(directory, LABEL_RED)
    y_filenames, y_labels = fl_for_directory_and_label(directory, LABEL_YELLOW)
    g_filenames, g_labels = fl_for_directory_and_label(directory, LABEL_GREEN)
    all_filenames = r_filenames + y_filenames + g_filenames
    all_labels = r_labels + y_labels + g_labels
    return (all_filenames, all_labels)

def shuffled_fl_for_fl(filenames, labels):
    merged_fl = list(zip(filenames, labels))
    random.shuffle(merged_fl)
    shuffled_filenames, shuffled_labels = zip(*merged_fl)
    return (shuffled_filenames, shuffled_labels)

def split_name_for_idx(idx):
    if idx < num_train_samples - 1:
        return SPLIT_TRAIN
    return SPLIT_VAL

def copy_fl_to_output_dir(filenames, labels):
    assert len(filenames) == len(labels)
    for idx in range(len(filenames)):
        split_dirname = split_name_for_idx(idx)
        filename = filenames[idx]
        label = labels[idx]
        file_path = os.path.join(input_dir, label, filename)
        label_split_output_dir = os.path.join(output_dir, split_dirname, label)
        shutil.copy2(file_path, label_split_output_dir)

def main():
    create_output_dirs()
    filenames, labels = data_for_directory(input_dir)
    shuffled_filenames, shuffled_labels = shuffled_fl_for_fl(filenames, labels)
    copy_fl_to_output_dir(shuffled_filenames, shuffled_labels)

if __name__ == '__main__':
    main()
