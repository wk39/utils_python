import os
import random
import cv2

class Anno:
    def __init__(self):
        self.corners    = None
        self.rect       = None
        self.dn_rect    = None
        self.occluded   = None
        self.name1      = ''
        self.num1       = None
        self.name2      = ''

class AnnoDb:

    def __init__(self):
        self.anno_dict = {}
        self.class_list = []


    def read_udacity_dataset(self, img_dir, label_file):

        W,H = None,None

        self.image_dir = os.path.abspath(os.path.expanduser(img_dir))
        self.root_dir  = os.path.abspath(os.path.expanduser(os.path.join(self.image_dir,'..')))
        # self.anno_dir  = os.path.abspath(os.path.join(self.root_dir,'labels'))
        self.anno_dir  = self.image_dir

        label_file     = os.path.abspath(label_file)

        with open(label_file) as f:
            lines = f.readlines()

            for line in lines:
                sp = line.strip().split(',')
                # print(line)
                # print(sp)

                if len(sp)==7:
                    fname, sx1,sy1,sx2,sy2, soc, cls = sp
                    x1 = int(sx1)
                    y1 = int(sy1)
                    x2 = int(sx2)
                    y2 = int(sy2)
                    oc = int(soc)
                    #
                    cls = cls.replace('"','')
                    sub = ''

                elif len(sp)==8:
                    fname, sx1,sy1,sx2,sy2, soc, cls, sub = sp
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
                    img = cv2.imread(os.path.join(img_dir,fname))
                    H, W, D = img.shape

                # init if new...
                if fname not in self.anno_dict:
                    self.anno_dict[fname] = []

                # add if new class
                if cls not in self.class_list:
                    self.class_list.append(cls)

                #
                an = Anno()
                an.corners    = (x1,y1,x2,y2)
                an.rect       = (x1,y1,x2-x1,y2-y1)
                an.dn_rect    = ((x1+x2)/2/W,(y1+y2)/2/H,(x2-x1)/W,(y2-y1)/H)
                an.occluded   = oc
                an.name1      = cls
                an.num1       = self.class_list.index(cls)
                an.name2      = sub

                # add 
                self.anno_dict[fname].append(an)


    def write_darknet_txt(self):

        if not os.path.exists(self.anno_dir):
            os.mkdir(self.anno_dir)

        # labels
        for fname, ans in self.anno_dict.items():
            fname_full = os.path.join(self.anno_dir, fname.split('.')[0]+'.txt')
            with open(fname_full, 'wt') as f:
                for an in ans:
                    f.write('%d %f %f %f %f\n'%(an.num1, *an.dn_rect))
            f.close()
        # names
        with open(os.path.join(self.root_dir,'names.txt'), 'wt') as f:
            for cl in self.class_list:
                f.write('%s\n'%(cl))
            f.close()


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

    d = './object-dataset'
    f = './object-dataset/label.csv'

    adb.read_udacity_dataset(d, f)
    print('Annotation DB is ready (%d objects)'%(len(adb.anno_dict)))
    adb.write_darknet_txt()
    print(' writing darknet text files ... done.')


