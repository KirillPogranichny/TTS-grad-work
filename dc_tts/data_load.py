from __future__ import print_function

import tokenizer
from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata


def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char


def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode == "train":
        # Parse
        fpaths, text_lengths, texts = [], [], []
        # transcript = os.path.join(hp.trainscript)
        lines = codecs.open(hp.transcript, 'r', 'utf-8').readlines()
        for line in lines:
            fname, _, text, _ = line.strip().split("|")

            fpath = os.path.join(hp.data, fname)
            fpaths.append(fpath)

            text += u"␃"  # ␃: EOS
            text = [char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())

        return fpaths, text_lengths, texts
    else: # synthesize on unseen test text.
        # Parse
        def _normalize(line):
            text = line.split("|")[0]
            text = " ".join(text.split(" ")[1:])
            text += u"␃"
            return text
        lines = codecs.open(hp.test_data, 'r', 'utf-8').read().splitlines()
        sents = [_normalize(line) for line in lines[1:]] # ␃: EOS
        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            print(sent)
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts


def get_batch():
    """Loads training data and put them in queues"""
    # Load data
    fpaths, text_lengths, texts = load_data() # list
    maxlen, minlen = max(text_lengths), min(text_lengths)

    # Create Dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((fpaths, text_lengths, texts))

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(fpaths)).batch(hp.B)

    # Create iterator
    iterator = iter(dataset)

    # Get next batch
    fpath, text_length, text = iterator.get_next()

    # Pad text to the maximum length within the batch
    max_text_length = tf.reduce_max(text_length)
    text = tf.strings.substr(text, 0, max_text_length)  # Extract substrings of fixed length

    # Parse
    text = tf.io.decode_raw(text, tf.int32)  # (None,)

    if hp.prepro:
        def _load_spectrograms(fpath):
            fname = os.path.basename(fpath)
            mel = "{}/mels/{}".format("/data/private/multi-speech-corpora/dc_tts/"+hp.lang,
                                      fname.replace("wav", "npy"))
            mag = "{}/mags/{}".format("/data/private/multi-speech-corpora/dc_tts/"+hp.lang,
                                      fname.replace("wav", "npy"))
            return fname, np.load(mel), np.load(mag)

        fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
    else:
        fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

    # Add shape information
    fname.set_shape(())
    text.set_shape((None,))
    mel.set_shape((None, hp.n_mels))
    mag.set_shape((None, hp.n_fft//2+1))

    return text, mel, mag, fname, len(fpaths) // hp.B

