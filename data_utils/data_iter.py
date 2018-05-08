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

class ImageRecIterLstm(mx.io.DataIter):

    def __init__(self, prefix, batch_size, data_shape, num_label, lstm_init_states, shuffle=False, name=None
        , last_batch_handle='pad'):

        super(ImageRecIterLstm, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label

        self.init_states = lstm_init_states
        if self.init_states is not None:
            self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]

        self.record = mx.recordio.MXIndexedRecordIO(prefix+'.idx', prefix+'.rec', 'r')
        
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))]
        if self.init_states is not None:
            self.provide_data += lstm_init_states
        self.provide_label = [('label', (self.batch_size, self.num_label))]
        self.name = name

        with open(prefix+'.idx') as f:
            self.num_data = len(f.readlines())
        
        self.shuffle = shuffle
        self.idx = list(range(0, self.num_data))
        if self.shuffle:
            random.shuffle(self.idx)

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
            if self.init_states is not None:
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
            data_all = [mx.nd.array(data)]
            if self.init_states is not None:
                data_all += self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            if self.init_states is not None:
                data_names += init_state_names
            label_names = ['label']
            return SimpleBatch(data_names, data_all, label_names, label_all)
        else:
            raise StopIteration