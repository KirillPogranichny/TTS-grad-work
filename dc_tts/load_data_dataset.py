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


def bucket_batch_dataset(dataset, batch_size, bucket_boundaries, padded_shapes, padding_values):
    # Создаем пакеты данных с использованием bucket_by_sequence_length
    return dataset.experimental.bucket_by_sequence_length(
        lambda fname, mel, mag, text_length, text: text_length,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=[batch_size] * (len(bucket_boundaries) + 1),
        padded_shapes=padded_shapes,
        padding_values=padding_values
    )


def get_batch():
    fpaths, text_lengths, texts = load_data()

    maxlen, minlen = max(text_lengths), min(text_lengths)
    num_batch = len(fpaths) // hp.B

    dataset = tf.data.Dataset.from_tensor_slices((fpaths, text_lengths, texts))
    dataset = dataset.shuffle(buffer_size=len(fpaths))
    dataset = dataset.map(lambda fpath, text_length, text:
                          (fpath, text_length, text_to_int(text)))

    if hp.prepro:

        def _load_spectrograms(fpath, text_length, text):
            fname = os.path.basename(fpath)
            mel = f"mels/{fname.replace("wav", "npy")}"
            mag = f"mags/{fname.replace("wav", "npy")}"
            return fname, mel, mag, text_length, text

        dataset = dataset.map(lambda fpath, text_length, text:
                              tf.py_function(_load_spectrograms, [fpath.numpy().decode('utf-8'), text_length, text],
                                             [tf.string, tf.float32, tf.float32, tf.int32, tf.int32]))
    else:
        dataset = dataset.map(lambda fpath, text_length, text:
                              tf.py_function(load_spectrograms, [fpath.numpy().decode('utf-8'), text_length, text],
                                             [tf.string, tf.float32, tf.float32, tf.int32, tf.int32]))

    # dataset = dataset.map(lambda fname, mel, mag, text_length, text:
    #                       (fname, mel, mag, text_length, text,
    #                        tf.ensure_shape(fname, ()), tf.ensure_shape(mel, (None, hp.n_mels)),
    #                        tf.ensure_shape(mag, (None, hp.n_fft // 2 + 1)), tf.ensure_shape(mag, (None,))))

    bucket_boundaries = [i for i in range(minlen + 1, maxlen - 1, 20)]

    # Определяем формы пакетов данных
    padded_shapes = ((), (None, hp.n_mels), (None, hp.n_fft // 2 + 1), (), ())
    padding_values = (
        '', tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32),
        tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32))

    dataset = bucket_batch_dataset(dataset, hp.B, bucket_boundaries, padded_shapes, padding_values)

    return dataset, num_batch
