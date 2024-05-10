import os
import codecs
import numpy as np
from torch.utils.data import Dataset

vocab = "PE абвгдеёжзийклмнопрстуфхцчшщъыьэюя-.,?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


'''Необходимо провести нормализацию для русского языка'''
def text_normalize(text):
    text = text.lower()
    for c in "-—:":
        text = text.replace(c, "-")
    for c in "()\"«»“”';":
        text = text.replace(c, ",")
    for c in "!":
        text = text.replace(c, ".")
    return text


def _normalize(s):
    """remove leading '-'"""
    s = s.strip()
    if s[0] == '—' or s[0] == '-':
        s = s[1:].strip()
    s = s.replace('—', '') if '-' in s else s
    s = s.replace(':', ',') if ':' in s else s
    return s


def read_metadata(metadata_file):
    fnames, text_lengths, texts = [], [], []
    transcript = os.path.join(metadata_file)
    lines = codecs.open(transcript, 'rb', 'utf-8').readlines()
    for line in lines:
        fname, _, text, _ = line.strip().split("|")
        fname = fname.split('/')[-1]
        fname = fname.rsplit('.wav', 1)[0]
        fnames.append(fname)

        # text = _normalize(text)
        text = text_normalize(text) + "E"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.longlong))

    return fnames, text_lengths, texts


def get_test_data(sentences, max_n):
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentences]  # text normalization, E: EOS
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.longlong)
    for i, sent in enumerate(normalized_sentences):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts


class RUSpeech(Dataset):
    def __init__(self, keys, dir_name='RUSpeechDataset'):
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        self.fnames, self.text_lengths, self.texts = read_metadata(os.path.join(self.path, 'transcript.txt'))

    def slice(self, start, end):
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        available_keys = ['texts', 'mels', 'mags', 'mel_gates', 'mag_gates']

        for key in available_keys:
            if key in self.keys:
                if key == 'texts':
                    data[key] = self.texts[index]
                elif key == 'mels':
                    data[key] = np.load(os.path.join(self.path, 'mels', f'{self.fnames[index]}.npy'))
                elif key == 'mags':
                    data[key] = np.load(os.path.join(self.path, 'mags', f'{self.fnames[index]}.npy'))
                elif key == 'mel_gates':
                    data[key] = np.ones(data['mels'].shape[0], dtype=np.int32)
                elif key == 'mag_gates':
                    data[key] = np.ones(data['mags'].shape[0], dtype=np.int32)

        return data
