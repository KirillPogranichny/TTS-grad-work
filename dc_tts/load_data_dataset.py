from __future__ import print_function

from tensorflow.python.ops import array_ops
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


def get_batch():
    fpaths, text_lengths, texts = load_data()

    maxlen, minlen = max(text_lengths), min(text_lengths)
    num_batch = len(fpaths) // hp.B

    dataset = tf.data.Dataset.from_tensor_slices((fpaths, text_lengths, texts))
    # for fpath, text_length, text in dataset.take(1):
    #     print(text)
    dataset = dataset.shuffle(buffer_size=len(fpaths))
    # for fpath, text_length, text in dataset.take(1):
    #     print(text)
    dataset = dataset.map(lambda fpath, text_length, text:
                          (fpath, text_length, tf.io.decode_raw(text, tf.int32)))
    # for fpath, text_length, text in dataset.take(1):
    #     print(fpath)
    #     print(text)

    if hp.prepro:

        def _load_spectrograms(fpath, text_length, text):
            fname = os.path.basename(fpath)
            mel = f"mels/{fname.replace('wav', 'npy')}"
            mag = f"mags/{fname.replace('wav', 'npy')}"
            return fname, mel, mag, text_length, text

        dataset = dataset.map(lambda fpath, text_length, text:
                              tf.py_function(lambda f, tl, t: _load_spectrograms(f.numpy().decode('utf-8'), tl, t),
                                             [fpath, text_length, text],
                                             [tf.string, tf.float32, tf.float32, tf.int32, tf.int32]))
    else:
        dataset = dataset.map(lambda fpath, text_length, text:
                              tf.py_function(lambda f, tl, t: load_spectrograms(f.numpy().decode('utf-8'), tl, t),
                                             [fpath, text_length, text],
                                             [tf.string, tf.float32, tf.float32, tf.int32, tf.int32]))

    dataset = dataset.map(lambda fname, mel, mag, text_length, text:
                          (tf.ensure_shape(fname, ()),
                           tf.ensure_shape(mel, (None, hp.n_mels)),
                           tf.ensure_shape(mag, (None, hp.n_fft // 2 + 1)),
                           tf.ensure_shape(text_length, ()),
                           tf.ensure_shape(text, (None,))))

    bucket_boundaries = [i for i in range(minlen + 1, maxlen - 1, 20)]

    # Определяем формы пакетов данных
    padded_shapes = ((), (None, hp.n_mels), (None, hp.n_fft // 2 + 1), (), (None,))
    padding_values = (
        '', tf.constant(0, dtype=tf.float32),
        tf.constant(0, dtype=tf.float32),
        tf.constant(0, dtype=tf.int32),
        tf.constant(0, dtype=tf.int32))

    def _element_length_fn(x):
        # print("Shape of x:", array_ops.shape(x))
        return array_ops.shape(x)[0]

    def batch_dataset(dataset, batch, bucket_boundaries, padded_shapes, padding_values):
        # Создаем пакеты данных с использованием bucket_by_sequence_length
        return dataset.bucket_by_sequence_length(
            element_length_func=lambda *args: _element_length_fn(args[4]),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=[batch] * (len(bucket_boundaries) + 1),
            # padded_shapes=padded_shapes
            padding_values=padding_values
        )

    for fname, mel, mag, text_length, text in dataset.take(1):
        print('Before\n', fname, mel, mag, text_length, text)

    dataset = batch_dataset(dataset, hp.B, bucket_boundaries, padded_shapes, padding_values)

    for fname, mel, mag, text_length, text in dataset.take(1):
        print('After\n', fname, mel, mag, text_length, text)

    return (dataset.map(lambda fname, mel, mag, text_length, text: text),
            dataset.map(lambda fname, mel, mag, text_length, text: mel),
            dataset.map(lambda fname, mel, mag, text_length, text: mag),
            dataset.map(lambda fname, mel, mag, text_length, text: fname),
            num_batch)
