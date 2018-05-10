from __future__ import print_function

from dotdict import DotDict

hp = DotDict()

# Training hyper parameters
hp.train_epoch_size = 30000
hp.eval_epoch_size = 3000
hp.num_epoch = 20
hp.learning_rate = 0.02
hp.momentum = 0.9
hp.bn_mom = 0.9
hp.workspace = 512
hp.loss_type = "warpctc" # ["warpctc"  "ctc"]

hp.batch_size = 1024
hp.num_classes = 5990 # 0 as blank, 1~xxxx as labels
hp.img_width = 280
hp.img_height = 32

# LSTM hyper parameters
hp.num_hidden = 100
hp.num_lstm_layer = 2
hp.seq_length = 35
hp.num_label = 10
hp.dropout = 0.5

