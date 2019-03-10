from nn.annotation_database import AnnoDb

import h5py
import numpy as np
import cv2
import os

import argparse

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

argparser.add_argument(
    '-s',
    '--split',
    help="number files to split",
    default='1')


if __name__ == '__main__':

    args = argparser.parse_args()
    #
    path_dataset = os.path.expanduser(args.data_path)
    path_label   = os.path.expanduser(args.label_path)
    filetype     = args.format
    dataset_name = args.name
    nsplit       = int(args.split)

    print(args)
    #print(exception)

    adb = AnnoDb()


    adb.read_udacity_dataset(path_dataset, path_label)
    print('Annotation DB is ready (%d objects)'%(len(adb.anno_dict)))
    # adb.write_darknet_txt()
    # print(' writing darknet text files ... done.')

    adb.shuffle()   # IMPORTANT

    # save h5 files
    # SCALE = 0.5
    ntotal = adb.get_size()
    npart  = ntotal//nsplit
    H,W,D = adb.get_image_size()
    # NH, NW = int(H*SCALE), int(W*SCALE)
    NH, NW = 416,416
    for i in range(nsplit):
        print('loading image data & labels... %d/%d'%(i+1,nsplit))
        images = np.zeros((npart,NH,NW,D), dtype=np.float32)
        labels = np.zeros((npart,adb.get_max_objects(),5), dtype=np.float32)
        ofs = i*npart
        for j in range(npart):
            an = adb.anno_list[ofs+j]
            images[j] = cv2.resize(
                    cv2.cvtColor( cv2.imread(an.fname_full), cv2.COLOR_BGR2RGB),
                    (NW,NH) ).astype(np.float32)/255.
            for jj, obj in enumerate(an.objects):
                labels[j][jj] = [
                        obj.bbox_darknet[0],
                        obj.bbox_darknet[1],
                        obj.bbox_darknet[2],
                        obj.bbox_darknet[3],
                        obj.num1
                        ]
            if j%500==0:
                print(' ... %d/%d - %d processed'%(i+1,nsplit,j))
        print(' done.')
        
        print('saving %s file (%d/%d)...'%(filetype, i+1,nsplit))
        if nsplit>1:
            fname = dataset_name + '_%d.'%(i) + filetype
        else:
            fname = dataset_name + '.' + filetype

        if filetype=='npz':
            np.savez(os.path.join(adb.root_dir, fname),
                images = images,
                boxes = labels)
        elif filetype=='h5':
            f = h5py.File(os.path.join(adb.root_dir, fname),'w')
            x = f.create_dataset('images', data=images)
            y = f.create_dataset('boxes' , data=labels)
            f.close()
        print(' done.')
        
    # class names
    adb.write_class_text(dataset_name+'.txt')


