# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import mxnet as mx
import cv2

prefix = '/mnt/15F1B72E1A7798FD/DK2/ocr_rec/val'
record = mx.recordio.MXIndexedRecordIO(prefix+'.idx', prefix+'.rec', 'r')

with open(prefix+'.idx') as f:
    num_data = len(f.readlines())

with open(b'/mnt/6B133E147DED759E/Synthetic Chinese String Dataset/char_std_5990.txt') as fchar:
    charset = [_.decode('gb18030').strip() for _ in fchar.readlines()]

for il in range(num_data):
    pack = record.read_idx(il)        
    header, img = mx.recordio.unpack_img(pack)
    # img = np.expand_dims(img, axis=0)
    label = header.label
    label = label.astype(np.int32)
    cv2.imshow("w", img)
    print( ''.join([charset[_].encode('utf8') for _ in label]) )

    cv2.waitKey()