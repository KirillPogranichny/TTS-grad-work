import os
import re
import sys
import argparse
from tqdm import tqdm
import gdown

import numpy as np
import torch
import asyncio

from models.text2mel import Text2Mel
from models.ssrn import SSRN
from hparams import HParams as hp
from audio import save_to_wav
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png


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


def download_model_from_drive(drive_url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading model from {drive_url} to {output_path}...")
        gdown.download(drive_url, output_path, quiet=False)
        # Проверка размера файла после скачивания
        if os.path.getsize(output_path) < 1024 * 1024:  # Меньше 1 МБ - скорее всего, ошибка
            os.remove(output_path)
            raise ValueError(f"Error downloading model from {drive_url}. File size is too small.")


def synthesize(sentences, dataset='ruspeech'):
    if dataset == 'ljspeech':
        from datasets.lj_speech import vocab, get_test_data
        samples_path = os.path.join('dnn_tts_torch/samples', 'en')
        text2mel_drive_url = 'https://drive.google.com/uc?export=download&id=1E7_1pwkM2uAawSY0uc1bcZ0S3LUbU9mO'
        ssrn_drive_url = 'https://drive.google.com/uc?export=download&id=1ORrzqHfcRc8GjLkoBEtDJpOpYRhCWN8K'
    else:
        from datasets.ru_speech import vocab, get_test_data
        samples_path = os.path.join('dnn_tts_torch/samples', 'ru')
        text2mel_drive_url = 'https://drive.google.com/uc?export=download&id=1Amo3CpUaYMPloVdNcu0Bbfj6gFKhkDLl'
        ssrn_drive_url = 'https://drive.google.com/uc?export=download&id=1YF-9TRdmd_U2a-8RULkya4iMfyrG2nE0'

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
        os.path.join(hp.logdir, f'{dataset}-text2mel'))

    print(f"last_checkpoint_file_name for text2mel: {last_checkpoint_file_name}")

    if not last_checkpoint_file_name:
        last_checkpoint_file_name = os.path.join(hp.logdir, f'{dataset}-text2mel.pth')
        download_model_from_drive(text2mel_drive_url, last_checkpoint_file_name)

    try:
        print(f"loading text2mel checkpoint '{last_checkpoint_file_name}'...")
        load_checkpoint(last_checkpoint_file_name, text2mel, None)
    except Exception as e:
        print(f"Failed to load text2mel checkpoint: {e}")
        sys.exit(1)

    ssrn = SSRN().eval()
    last_checkpoint_file_name = get_last_checkpoint_file_name(
        os.path.join(hp.logdir, f'{dataset}-ssrn'))

    print(f"last_checkpoint_file_name for ssrn: {last_checkpoint_file_name}")

    if not last_checkpoint_file_name:
        last_checkpoint_file_name = os.path.join(hp.logdir, f'{dataset}-ssrn.pth')
        download_model_from_drive(ssrn_drive_url, last_checkpoint_file_name)

    try:
        print(f"loading ssrn checkpoint '{last_checkpoint_file_name}'...")
        load_checkpoint(last_checkpoint_file_name, ssrn, None)
    except Exception as e:
        print(f"Failed to load ssrn checkpoint: {e}")
        sys.exit(1)

    for i in range(len(sentences)):
        sentence = [sentences[i]]
        max_N = len(sentences[i])
        L = torch.from_numpy(get_test_data(sentence, max_N))
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

        save_to_png(f'{samples_path}/{max_number + i + 1}-att.png', A[0, :, :])
        save_to_png(f'{samples_path}/{max_number + i + 1}-mel.png', Y[0, :, :])
        save_to_png(f'{samples_path}/{max_number + i + 1}-mag.png', Z[0, :, :])
        save_to_wav(Z[0, :, :].T, f'{samples_path}/{max_number + i + 1}-wav.wav')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d',
        "--dataset",
        required=False,
        choices=['ljspeech', 'ruspeech'],
        default='ruspeech',
        help='dataset name (default: ruspeech)')
    parser.add_argument(
        '-t',
        "--text",
        type=str,
        required=False,
        help="Text to synthesize")
    args = parser.parse_args()

    if args.text:
        sentences = [args.text]
    else:
        sentences = asyncio.run(enter_sentences())

    synthesize(sentences, args.dataset)


if __name__ == "__main__":
    main()
