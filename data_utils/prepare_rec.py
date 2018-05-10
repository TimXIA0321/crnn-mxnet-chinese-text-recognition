
import os
import sys
import numpy as np
import mxnet as mx
import cv2

num_label=10
data_shape = (280, 32)
data_root = '/mnt/6B133E147DED759E/Synthetic Chinese String Dataset/images'
train_fn='/mnt/6B133E147DED759E/Synthetic Chinese String Dataset/train.txt'
val_fn='/mnt/6B133E147DED759E/Synthetic Chinese String Dataset/test.txt'

def make_rec(fn, prefix):
    lines = open(fn).readlines()
    record = mx.recordio.MXIndexedRecordIO(prefix+'.idx', prefix+'.rec', 'w')


    for i, l in enumerate(lines):
        img_lst = l.strip().split(' ')
        img_path = os.path.join(data_root, img_lst[0])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, data_shape,interpolation=cv2.INTER_LINEAR)

        ret = np.zeros(num_label, np.int32)
        for idx in range(1, len(img_lst)):
            ret[idx - 1] = int(img_lst[idx])

        # TODO: change to png
        p = mx.recordio.pack_img((0,ret,i,0), img, quality=0, img_fmt='.png')        
        record.write_idx(i,p)
        i += 1

    record.close()   

make_rec(train_fn, 'train')
make_rec(val_fn, 'val')