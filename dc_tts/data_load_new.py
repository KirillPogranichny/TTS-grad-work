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


def load_data(mode='train'):
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode == 'train':
        fpaths, text_lengths, texts = [], [], []
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

    else:
        # synthesize on unseen test text.
        def _normalize(line):
            text = line.split("|")[0]
            text = " ".join(text.split(" ")[1:])
            text += u"␃"
            return text

        lines = codecs.open(hp.test_data, 'r', 'utf-8').read().splitlines()
        sents = [_normalize(line) for line in lines[1:]]  # ␃: EOS
        texts = np.zeros((len(sents), hp.max_N), np.int32)
        for i, sent in enumerate(sents):
            print(sent)
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        return texts


def decode_dataset(sequence):
    return ' '.join(tokenizer.index_word[idx] for idx in sequence.numpy() if idx != 0)


def process_data(fpath, text_length, text):
    decoded_text = decode_dataset(text)
    return fpath, text_length, decoded_text


def put_texts(dataset):
    text = dataset.map(lambda fpath, text_length, decoded_text: decoded_text)
    return text


def get_batch():
    with tf.device('/cpu:0'):
        fpaths, text_lengths, texts = load_data()
        maxlen, minlen = max(text_lengths), min(text_lengths)

        num_batch = len(fpaths) // hp.B

        dataset = tf.data.Dataset.from_tensor_slices((fpaths, text_lengths, texts))
        dataset = dataset.shuffle(buffer_size=len(texts))
        print(f'dataset:\n{dataset}\n')
        ctr = 0
        for fpaths, text_lengths, texts in dataset:
            if ctr == 1:
                break
            ctr += 1
            print(fpaths, text_lengths, texts)



        dataset = dataset.map(
            lambda f, t_len, t: tuple(tf.py_function(process_data, [f, t_len, t], [tf.string, tf.int32, tf.string])))
        print(f'dataset after processing:\n{dataset}')
        ctr = 0
        for fpaths, text_lengths, texts in dataset:
            if ctr == 1:
                break
            ctr += 1
            print(fpaths, text_lengths, texts)

        if hp.prepro:

            def _load_spectrograms(fpath_tensor):
                fpath = tf.strings.bytes_split(fpath_tensor).values[0].numpy().decode('utf-8')
                fname = os.path.basename(fpath)
                mel = "{}/mels/{}".format("/data/private/multi-speech-corpora/dc_tts/" + hp.lang,
                                          fname.replace("wav", "npy"))
                mag = "{}/mags/{}".format("/data/private/multi-speech-corpora/dc_tts/" + hp.lang,
                                          fname.replace("wav", "npy"))
                return fname, mel, mag

            results = list(dataset.map(lambda f, _, __: _load_spectrograms(f)).as_numpy_iterator())
        else:
            results = list(dataset.map(lambda f, _, __: load_spectrograms(f)).as_numpy_iterator())

        fname, mel, mag = zip(*results)

    text = put_texts(dataset)

    text.set_shape((None,))
    fname.set_shape(())
    mel.set_shape((None, hp.n_mels))
    mag.set_shape((None, hp.n_fft // 2 + 1))

    # Batching
    dataset = tf.data.Dataset.from_tensor_slices((text, fname, mel, mag))
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            lambda text, fname, mel, mag: mel.shape[0],
            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
            bucket_batch_sizes=[hp.B] * (num_batch // hp.B)
        ))

    iterator = dataset.make_one_shot_iterator()
    text, fname, mel, mag = iterator.get_next()

    return text, fname, mel, mag, num_batch


text, fname, mel, mag, num_batch = get_batch()
# print
