import os
import random
import cv2

class Obj:
    def __init__(self):
        self.bbox         = None
        self.rect         = None
        self.bbox_darknet = None
        self.occluded     = None
        self.name1        = ''
        self.num1         = None
        self.name2        = ''

class Anno:
    def __init__(self):
        self.fname      = ''
        self.fname_full = ''
        self.objects    = []

class AnnoDb:

    def __init__(self):
        self.anno_dict = {}
        self.class_list = []
        self.anno_list = []         # able to shuffle


    def shuffle(self,seed=None):

        print('suffling...', end='')
        random.seed(seed)
        random.shuffle(self.anno_list)
        print('done.')

    def get_max_objects(self):

        n = 0
        for an in self.anno_list:
            if len(an.objects)>n:
                n = len(an.objects)

        return n

    def get_image_size(self, index=0):

        img = cv2.imread(self.anno_list[index].fname_full)
        return img.shape

    def get_size(self):
        return len(self.anno_list)

    def get_class_names(self):
        return self.class_list


    def read_udacity_dataset(self, img_dir, label_file):

        print('reading udacity annotations...', end='')

        W,H = None,None

        self.image_dir = os.path.abspath(os.path.expanduser(img_dir))
        self.root_dir  = os.path.abspath(os.path.expanduser(os.path.join(self.image_dir,'..')))
        # self.anno_dir  = os.path.abspath(os.path.join(self.root_dir,'labels'))
        self.anno_dir  = self.image_dir

        label_file     = os.path.abspath(os.path.expanduser(label_file))

        with open(label_file) as f:
            lines = f.readlines()

            for line in lines:
                sp = line.strip().split(',')
                # print(line)
                # print(sp)

                if len(sp)==7:
                    img_name, sx1,sy1,sx2,sy2, soc, cls = sp
                    x1 = int(sx1)
                    y1 = int(sy1)
                    x2 = int(sx2)
                    y2 = int(sy2)
                    oc = int(soc)
                    #
                    cls = cls.replace('"','')
                    sub = ''

                elif len(sp)==8:
                    img_name, sx1,sy1,sx2,sy2, soc, cls, sub = sp
                    x1 = int(sx1)
                    y1 = int(sy1)
                    x2 = int(sx2)
                    y2 = int(sy2)
                    oc = int(soc)

                    #
                    cls = cls.replace('"','')
                    sub = sub.replace('"','')
                    ### FIXME
                    ### FIXME
                    ### FIXME
                    # cls = cls+'_'+sub

                else:
                    print('parsing error', sp)


                if W is None:
                    img = cv2.imread(os.path.join(self.image_dir,img_name))
                    H, W, D = img.shape

                # add new
                if img_name not in self.anno_dict:
                    an = Anno()
                    an.fname = img_name
                    an.fname_full = os.path.join(self.image_dir, img_name)
                    #
                    self.anno_dict[img_name] = an

                # update class list
                if cls not in self.class_list:
                    self.class_list.append(cls)

                # add object
                obj = Obj()
                obj.bbox         = (x1,y1,x2,y2)
                obj.rect         = (x1,y1,x2-x1,y2-y1)
                obj.bbox_darknet = ((x1+x2)/2/W,(y1+y2)/2/H,(x2-x1)/W,(y2-y1)/H)
                obj.occluded     = oc
                obj.name1        = cls
                obj.num1         = self.class_list.index(cls)
                obj.name2        = sub
                #
                self.anno_dict[img_name].objects.append(obj)

        # dictionary to list for shuffe
        self.anno_list = list(self.anno_dict.values())

        print('done. (', len(self.anno_list), 'annotations loaded.)')

    def write_class_text(self, fname='names.txt'):
        # class names
        with open(os.path.join(self.root_dir,fname), 'wt') as f:
            for cl in self.class_list:
                f.write('%s\n'%(cl))
            f.close()

    def write_darknet_txt(self):

        if not os.path.exists(self.anno_dir):
            os.mkdir(self.anno_dir)

        # labels
        for fname, an in self.anno_dict.items():
            fname_full = os.path.join(self.anno_dir, fname.split('.')[0]+'.txt')
            with open(fname_full, 'wt') as f:
                for obj in an.objects:
                    f.write('%d %f %f %f %f\n'%(obj.num1, *obj.bbox_darknet))
            f.close()

        # class names
        write_class_text()


        # train & valid list
        fnames = list(self.anno_dict.keys())
        #
        n = len(fnames)
        n_valid = n//100
        n_train = n-n_valid
        #
        random.shuffle(fnames)
        #
        fnames_train = fnames[0:n_train]
        fnames_valid = fnames[n_train:n]
        # print(fnames_train, n_train)
        # print(fnames_valid)
        # train list
        with open(os.path.join(self.root_dir,'train.txt'), 'wt') as f:
            for fname in sorted(fnames_train):
                f.write('%s\n'%(os.path.join(self.image_dir,fname)))
            f.close()
        # valid list
        with open(os.path.join(self.root_dir,'val.txt'), 'wt') as f:
            for fname in sorted(fnames_valid):
                f.write('%s\n'%(os.path.join(self.image_dir,fname)))
            f.close()



if __name__ == '__main__':

    adb = AnnoDb()

    d = '~/Data/udacity/object_detection/object-dataset'
    f = '~/Data/udacity/object_detection/labels_corrected.csv'

    adb.read_udacity_dataset(d, f)
    print('Annotation DB is ready (%d objects)'%(len(adb.anno_dict)))
    adb.write_darknet_txt()
    print(' writing darknet text files ... done.')


