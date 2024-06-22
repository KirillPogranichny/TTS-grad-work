import warnings
import os
import sys
import argparse
import subprocess

from audio import preprocess
from utils import download_file
from dnn_tts_torch.datasets.ru_speech import RUSpeech
from dnn_tts_torch.datasets.lj_speech import LJSpeech

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dataset",
    required=True,
    choices=[
        'ljspeech',
        'ruspeech'],
    help='dataset name')
args = parser.parse_args()

if args.dataset == 'ljspeech':
    dataset_file_name = 'LJSpeech-1.1.tar.bz2'
    datasets_path = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'datasets')
    dataset_path = os.path.join(datasets_path, 'LJSpeech-1.1')

    if os.path.isdir(dataset_path) and False:
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
    subprocess.run(['tar', 'xvjf', os.path.join(
        datasets_path, dataset_file_name)], check=True)

    print("Препроцессинг...")
    lj_speech = LJSpeech([])
    preprocess(dataset_path, lj_speech)

elif args.dataset == 'ruspeech':
    dataset_name = 'RUSpeechDataset'
    datasets_path = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'datasets')
    dataset_path = os.path.join(datasets_path, dataset_name)

    # Блок проверки целостности файлов русскоязычного датасета
    if os.path.isdir(dataset_path) and False:
        print("RUSpeech dataset уже существует")
        sys.exit(0)
    else:
        books = [
            'early_short_stories',
            'icemarch',
            'shortstories_childrenadults']
        for book_name in books:
            book_file_path = os.path.join(datasets_path, book_name)
            if not os.path.isfile(book_file_path):
                print(
                    f"'{book_name}' не найдена, пожалуйста, проверьте целостность файлов")
            else:
                print(f"'{book_name}' найден")

    dataset_transcript_file_name = 'transcript.txt'
    dataset_transcript_file_path = os.path.join(
        dataset_path, dataset_transcript_file_name)
    if not os.path.isfile(dataset_transcript_file_path):
        print(
            f"'{dataset_transcript_file_name}' не найден, пожалуйста, проверьте целостность файлов")
    else:
        print(f"'{dataset_transcript_file_name}' найден")

    # pre process
    print("pre processing...")
    ru_speech = RUSpeech([])
    preprocess(dataset_path, ru_speech)
