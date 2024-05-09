import os
import sys
import argparse
import fnmatch
import subprocess

from hparams import HParams as hp
from audio import preprocess
from utils import download_file
from datasets.ru_speech import RUSpeech
from datasets.lj_speech import LJSpeech

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['ljspeech', 'ruspeech'], help='dataset name')
args = parser.parse_args()

if args.dataset == 'ljspeech':
    dataset_file_name = 'LJSpeech-1.1.tar.bz2'
    datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    dataset_path = os.path.join(datasets_path, 'LJSpeech-1.1')

    if os.path.isdir(dataset_path):
        print("LJSpeech dataset уже существует")
        sys.exit(0)

    dataset_file_path = os.path.join(datasets_path, dataset_file_name)

    if not os.path.isfile(dataset_file_path):
        url = f"http://data.keithito.com/data/speech/{dataset_file_name}"
        download_file(url, dataset_file_path)
        print(f"Скачивание '{dataset_file_name}'...")
    else:
        print(f"'{dataset_file_name}' уже существует")

    print(f"Извлечение архива '{dataset_file_name}'...")
    subprocess.run(['tar', 'xvjf', os.path.join(datasets_path, dataset_file_name)], check=True)

    print("Препроцессинг...")
    lj_speech = LJSpeech([])
    preprocess(dataset_path, lj_speech)

elif args.dataset == 'ruspeech':
    dataset_name = 'RUSpeechDataset'
    datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    dataset_path = os.path.join(datasets_path, dataset_name)

    # Блок проверки целостности файлов русскоязычного датасета
    if os.path.isdir(dataset_path) and False:
        print("RUSpeech dataset folder already exists")
        sys.exit(0)
    else:
        books = ['early_short_stories', 'icemarch', 'shortstories_childrenadults']
        for book_name in books:
            book_file_path = os.path.join(datasets_path, book_name)
            if not os.path.isfile(book_file_path):
                print("'%s' не найдена, пожалуйста, проверьте целостность файлов" % book_name)
            else:
                print("'%s' уже существует" % book_name)

    dataset_transcript_file_name = 'transcript.txt'
    dataset_transcript_file_path = os.path.join(dataset_path, dataset_transcript_file_name)
    if not os.path.isfile(dataset_transcript_file_path):
        print("'%s' не найден, пожалуйста, проверьте целостность файлов" % dataset_transcript_file_name)
    else:
        print("'%s' уже существует" % dataset_transcript_file_name)

    sample_rate = 44100  # original sample rate
    total_duration_s = 0

    wavs_paths = [wavs_path for wavs_path in os.listdir(dataset_path) if not wavs_path.endswith('.txt')]
    for wavs_path in wavs_paths:
        if not os.path.isdir(wavs_path):
            print("'%s' не найдена, пожалуйста, проверьте целостность файлов" % wavs_path)


    def _normalize(s):
        """remove leading '-'"""
        s = s.strip()
        if s[0] == '—' or s[0] == '-':
            s = s[1:].strip()
        return s


    def _get_mp3_file(book_name, chapter):
        book_download_path = os.path.join(datasets_path, book_name)
        wildcard = "*%02d - DPI.mp3" % chapter
        for file_name in os.listdir(book_download_path):
            if fnmatch.fnmatch(file_name, wildcard):
                return os.path.join(book_download_path, file_name)
        return None

    # pre process
    print("pre processing...")
    ru_speech = RUSpeech([])
    preprocess(dataset_path, ru_speech)
