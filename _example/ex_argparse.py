import argparse
import os

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-n',
    '--name',
    help="name of dataset",
    default='udacity_dataset')

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to dataset",
    default='~/Data/udacity/object_detection/object-dataset')

argparser.add_argument(
    '-l',
    '--label_path',
    help="path to label of dataset",
    default='~/Data/udacity/object_detection/labels_corrected.csv')

argparser.add_argument(
    '-f',
    '--format',
    help="format of output ['h5','npz']",
    default='h5')


if __name__ == '__main__':

    args = argparser.parse_args()
    #
    dataset_name = args.name
    path_dataset = os.path.abspath(os.path.expanduser(args.data_path))
    path_label   = os.path.abspath(os.path.expanduser(args.label_path))
    filetype     = args.format

    print(args)

