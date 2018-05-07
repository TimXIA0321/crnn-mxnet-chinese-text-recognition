from __future__ import print_function

import os
import cv2
import numpy as np
import mxnet as mx
import random


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names=list(), label=list()):
        self._data = data
        self._label = label
        self._data_names = data_names
        self._label_names = label_names

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def data_names(self):
        return self._data_names

    @property
    def label_names(self):
        return self._label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self._data_names, self._data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self._label_names, self._label)]


class ImageIter(mx.io.DataIter):

    """
    Iterator class for generating captcha image data
    """
    def __init__(self, data_root, data_list, batch_size, data_shape, num_label, name=None):
        """
        Parameters
        ----------
        data_root: str
            root directory of images
        data_list: str
            a .txt file stores the image name and corresponding labels for each line
        batch_size: int
        name: str
        """
        super(ImageIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label  = num_label

        self.data_root = data_root
        self.dataset_lst_file = open(data_list)

        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))]
        self.provide_label = [('label', (self.batch_size, self.num_label))]
        self.name = name

    def __iter__(self):
        data = []
        label = []
        cnt = 0
        for m_line in self.dataset_lst_file:
            img_lst = m_line.strip().split(' ')
            img_path = os.path.join(self.data_root, img_lst[0])

            cnt += 1
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.data_shape,interpolation=cv2.INTER_LINEAR)
            img = np.expand_dims(img, axis=0)
            data.append(img)

            ret = np.zeros(self.num_label, np.int32)
            for idx in range(1, len(img_lst)):
                ret[idx-1] = int(img_lst[idx])

            label.append(ret)
            if cnt % self.batch_size == 0:
                data_all = [mx.nd.array(data)]
                label_all = [mx.nd.array(label)]
                data_names = ['data']
                label_names = ['label']
                data.clear()
                label.clear()
                yield SimpleBatch(data_names, data_all, label_names, label_all)
                continue


    def reset(self):
        if self.dataset_lst_file.seekable():
            self.dataset_lst_file.seek(0)

class ImageRecIterLstm(mx.io.DataIter):

    def __init__(self, prefix, batch_size, data_shape, num_label, lstm_init_states, name=None
        , last_batch_handle='pad'):

        super(ImageRecIterLstm, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label

        self.init_states = lstm_init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]

        self.record = mx.recordio.MXIndexedRecordIO(prefix+'.idx', prefix+'.rec', 'r')
        
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + lstm_init_states
        self.provide_label = [('label', (self.batch_size, self.num_label))]
        self.name = name

        with open(prefix+'.idx') as f:
            self.num_data = len(f.readlines())
        # self.shuffle = shuffle
        self.idx = list(range(0, self.num_data))
        # if shuffle:
        #     random.shuffle(self.idx)

        assert self.num_data >= batch_size, \
            "batch_size needs to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    def hard_reset(self):
        """Ignore roll over data and set to start."""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0
    
    def next(self):
        if self.iter_next():
            """Load data from underlying arrays, internal use only."""
            assert(self.cursor < self.num_data), "DataIter needs reset."
            if self.cursor + self.batch_size <= self.num_data:
                l = list(range(self.cursor, self.cursor + self.batch_size))
            else:
                pad = self.batch_size - self.num_data + self.cursor
                l = list(range(self.cursor, self.num_data)) + list(range(0,pad))
            
            init_state_names = [x[0] for x in self.init_states]
            data = []
            label = []

            for il in l:
                pack = self.record.read_idx(il)        
                header, img = mx.recordio.unpack_img(pack)
                img = np.expand_dims(img, axis=0)
                ret = header.label

                data.append(img)
                label.append(ret)
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']
            return SimpleBatch(data_names, data_all, label_names, label_all)
        else:
            raise StopIteration