from __future__ import print_function

from hyperparams import Hyperparams as hp
from utils import *
import codecs
import os


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
            texts.append(np.array(text, np.int32).tobytes())

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


def text_to_int(text):
    unicode_text = tf.strings.unicode_decode(text, input_encoding='UTF-8')
    # Преобразуем строки Unicode в числовые значения tf.int32
    int_text = tf.cast(unicode_text, tf.int32)
    return int_text


# def decode_bytes(byte_string):
#     decoded_string = np.array(byte_string.numpy().decode('utf-8'))
#     return decoded_string
#
# # Применяем функцию к каждому элементу тензора с использованием tf.py_function
# def decode_fpath(fpath):
#     return tf.py_function(decode_bytes, [fpath], tf.string)


def get_batch():
    fpaths, text_lengths, texts = load_data()
    # print(type(fpaths))
    # print(type(text_lengths))
    # print(type(text))
    dataset = tf.data.Dataset.from_tensor_slices((fpaths, text_lengths, texts))
    print(dataset)
    for fp, tl, t in dataset:
        print(fp, tl, t)
        break

    dataset = dataset.shuffle(buffer_size=len(fpaths))
    print('\n')
    print(dataset)
    for fp, tl, t in dataset:
        print(fp, tl, t)
        break

    dataset = dataset.map(lambda fpath, text_length, text:
                          (fpath, text_length, text_to_int(text)))
    print('\n')
    print(dataset)
    for fp, tl, t in dataset:
        print(fp, tl, t)
        break

    if hp.prepro:
        def _load_spectrograms(fpath):
            fname = os.path.basename(fpath)
            mel = "{}/mels/{}".format("/data/private/multi-speech-corpora/dc_tts/" + hp.lang,
                                      fname.replace("wav", "npy"))
            mag = "{}/mags/{}".format("/data/private/multi-speech-corpora/dc_tts/" + hp.lang,
                                      fname.replace("wav", "npy"))
            return fname, np.load(mel), np.load(mag)

        def process_fpath_prepro(fpath):
            fname, mel, mag = tf.py_function(_load_spectrograms,
                                             [fpath],
                                             [tf.string, tf.float32, tf.float32])
            return fname, mel, mag

        dataset = dataset.map(lambda fpath, text_length, text: process_fpath_prepro(fpath))
    else:
        def process_fpath(fpath):  # (None, n_mels)
            fname, mel, mag = tf.py_function(load_spectrograms,
                                             [fpath],
                                             [tf.string, tf.float32, tf.float32])
            return fname, mel, mag

        dataset = dataset.map(lambda fpath, text_length, text: process_fpath(fpath))


get_batch()
