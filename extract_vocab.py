from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import json

import misc.utils as utils

with open(os.path.join('/home/yz9244/AoANet/log/log_aoanet_rl', 'infos_'+'aoanet'+'.pkl'), 'rb') as f:
    infos = utils.pickle_load(f)
out = {}
out['ix_to_word'] = {}
for i in range(len(infos['vocab'].keys())): # encode the (1-indexed) vocab
    out['ix_to_word'][str(i+1)] = infos['vocab'][str(i+1)]

json.dump(out, open('/home/yz9244/AoANet/data/'+'cocotalk_vocab.json', 'w'))



