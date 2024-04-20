from __future__ import print_function

from utils import load_spectrograms
import os
from load_data_dataset import load_data
import numpy as np
import tqdm

# Load data
fpaths, _, _ = load_data() # list

for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag, _, _ = load_spectrograms(fpath, _, _)
    if not os.path.exists("mels"):
        os.mkdir("mels")
    if not os.path.exists("mags"):
        os.mkdir("mags")

    np.save("mels/{}".format(fname.replace("wav", "npy")), mag)
    np.save("mags/{}".format(fname.replace("wav", "npy")), mag)
