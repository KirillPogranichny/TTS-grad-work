import socket
import os
import sys
import re
import argparse
from tqdm import tqdm

import numpy as np
import torch
import asyncio

from models.text2mel import Text2Mel
from models.ssrn import SSRN
from hparams import HParams as hp
from audio import save_to_wav
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-d',
    "--dataset",
    required=True,
    choices=[
        'ljspeech',
        'ruspeech'],
    help='dataset name')
parser.add_argument(
    '-t',
    "--text",
    type=str,
    required=False,
    help="Text to synthesize")
args = parser.parse_args()


def extract_last_file(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    return 0


async def get_input():
    return input("Введите предложение для синтеза речи (введите 'q' после текста для выхода): ")


async def enter_sentences():
    SENTENCES = []
    while True:
        sentence = await get_input()
        if sentence.endswith(' q'):
            SENTENCES.append(sentence[:-2])
            break
        SENTENCES.append(sentence)
    return SENTENCES


if args.text:
    SENTENCES = [args.text]
else:
    SENTENCES = asyncio.run(enter_sentences())

# SENTENCES = []
# while True:
#     sentence = input("Введите предложение для синтеза речи (введите 'q' для выхода): ")
#     if sentence.lower() == 'q':
#         break
#     SENTENCES.append(sentence)


if args.dataset == 'ljspeech':
    from datasets.lj_speech import vocab, get_test_data

    samples_path = os.path.join('dc_tts_torch/samples', 'en')

else:
    from datasets.ru_speech import vocab, get_test_data

    samples_path = os.path.join('dc_tts_torch/samples', 'ru')

if not os.path.isdir(samples_path):
    os.mkdir(samples_path)
    max_number = 0
else:
    samples_path_list = os.listdir(samples_path)
    if not samples_path_list:
        max_number = 0
    else:
        max_number_file = max(samples_path_list, key=extract_last_file)
        max_number = extract_last_file(max_number_file)

torch.set_grad_enabled(False)

text2mel = Text2Mel(vocab).eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(
    os.path.join(hp.logdir, '%s-text2mel' % args.dataset))
if last_checkpoint_file_name:
    print("loading text2mel checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, text2mel, None)
else:
    print("text2mel not exits")
    sys.exit(1)

ssrn = SSRN().eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(
    os.path.join(hp.logdir, '%s-ssrn' % args.dataset))
if last_checkpoint_file_name:
    print("loading ssrn checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, ssrn, None)
else:
    print("ssrn not exits")
    sys.exit(1)

for i in range(len(SENTENCES)):
    sentences = [SENTENCES[i]]

    max_N = len(SENTENCES[i])
    L = torch.from_numpy(get_test_data(sentences, max_N))
    zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
    Y = zeros
    A = None

    for t in tqdm(range(hp.max_T)):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)

    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()

    save_to_png(f'{samples_path}/%d-att.png' %
                (max_number + i + 1), A[0, :, :])
    save_to_png(f'{samples_path}/%d-mel.png' %
                (max_number + i + 1), Y[0, :, :])
    save_to_png(f'{samples_path}/%d-mag.png' %
                (max_number + i + 1), Z[0, :, :])
    save_to_wav(Z[0, :, :].T,
                f'{samples_path}/%d-wav.wav' %
                (max_number + i + 1))
