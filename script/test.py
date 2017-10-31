from nn.annotation_database import AnnoDb

import h5py
import numpy as np
import cv2
import os

import argparse

import u_opencv as ucv

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

    adb = AnnoDb()


    adb.read_udacity_dataset(path_dataset, path_label)
    print('Annotation DB is ready (%d objects)'%(len(adb.anno_dict)))
    # adb.write_darknet_txt()
    # print(' writing darknet text files ... done.')

    adb.shuffle()   # IMPORTANT


    ntotal = adb.get_size()
    npart  = 50
    H,W,D = adb.get_image_size()
    # NH, NW = int(H*SCALE), int(W*SCALE)
    NH, NW = 416,416
    for i in range(nsplit):
        print('loading image data & labels... %d/%d'%(i+1,nsplit))
        #images = np.zeros((npart,NH,NW,D), dtype=np.float32)
        #labels = np.zeros((npart,adb.get_max_objects(),5), dtype=np.float32)
        #ofs = i*npart
        for j in range(npart):
            an = adb.anno_list[j]
            img = cv2.resize(
                    cv2.imread(an.fname_full),
                    (NW,NH) )

            for jj, obj in enumerate(an.objects):
                # draw boxes
                rcx, rcy, rw, rh = obj.bbox_darknet
                x = int((rcx-rw/2)*NW)
                y = int((rcy-rh/2)*NH)
                w = int(rw*NW)
                h = int(rh*NH)
                ucv.rectangle(img, (x,y,w,h), (0,0,255))


            ucv.label(img, an.fname, (0,NH) )

            cv2.imwrite('%d.jpg'%(j),img)

        print(' done.')
        
        
    # class names
    adb.write_class_text(dataset_name+'.txt')


