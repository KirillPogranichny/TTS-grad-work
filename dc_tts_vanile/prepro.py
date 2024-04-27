# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/css10
'''

from __future__ import print_function

from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
import tqdm
from hyperparams import Hyperparams as hp

# Load data
fpaths, _, _ = load_data() # list

for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag = load_spectrograms(fpath)
    if not os.path.exists("mels36"):
        os.mkdir("mels36")
    if not os.path.exists("mags36"):
        os.mkdir("mags36")

    np.save("mels36/{}".format(fname.replace("wav", "npy")), mag)
    np.save("mags36/{}".format(fname.replace("wav", "npy")), mag)