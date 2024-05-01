"""These methods are copied from https://github.com/Kyubyong/dc_tts/"""

import os
import copy
import librosa
import scipy.io.wavfile
import numpy as np
import shutil

from tqdm import tqdm
from scipy import signal
from hparams import HParams as hp
from datasets.ru_speech import RUSpeech


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag ** hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    '''Возвращаемые данные представляют собой временные ряды аудиоданных в виде чисел с плавающей точкой. 
       Это означает, что каждое значение в массиве представляет собой амплитуду звукового сигнала в определенный момент.
       Эти данные не имеют единиц измерения и не представляют давление звукового давления. 
       Вместо этого, они представляют абсолютную амплитуду звукового сигнала'''
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming:
    '''Аудиоданные обрезаются по началу и концу, чтобы удалить нежелательные шумы или тишину.'''
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    '''Применяется предварительное усиление, чтобы уменьшить высокочастотные компоненты сигнала,
       что помогает улучшить качество последующих преобразований.
       Это делается путем вычитания части предыдущего значения из каждого текущего значения,
       умноженного на коэффициент предварительного усиления hp.preemphasis'''
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    '''Выполняется преобразование Фурье в короткий промежуток времени (STFT).
       Это преобразование позволяет анализировать частотные характеристики аудиосигнала в различных временных окнах'''
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    '''Получение магнитудного спектрограммного представления, 
       которое показывает амплитудные характеристики аудиосигнала в частотной области'''
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    '''Сначала создается базис мел-фильтров с помощью librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels),
       а затем применяется к магнитудному спектрограммному представлению для получения мел-спектрограммы.
       Мел-спектрограмма используется для анализа аудиосигналов с точки зрения восприятия человеком'''
    mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    '''Магнитудные и мел-спектрограммы преобразуются в децибелы для улучшения визуализации и анализа'''
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    '''Спектрограммы нормализуются, чтобы их диапазон значений был одинаковым.
       Это делается для улучшения обучения и работы с нейронными сетями,
       так как они лучше работают с нормализованными данными'''
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    '''Спектрограммы переворачиваются, чтобы изменить формат с (частота, время) на (время, частота),
       что удобно для дальнейшей обработки и анализа'''
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def save_to_wav(mag, filename):
    """Generate and save an audio file from the given linear spectrogram using Griffin-Lim."""
    wav = spectrogram2wav(mag)
    scipy.io.wavfile.write(filename, hp.sr, wav)


def preprocess(dataset_path, speech_dataset):
    """Preprocess the given dataset."""
    wavs_path = os.path.join(dataset_path, 'wavs')

    if isinstance(speech_dataset, RUSpeech):
        if not os.path.isdir(wavs_path):
            os.mkdir(wavs_path)
        source_folders = [os.path.join(dataset_path, 'early_short_stories'),
                          os.path.join(dataset_path, 'icemarch'),
                          os.path.join(dataset_path, 'shortstories_childrenadults')]

        if any(os.path.isdir(source_folder) for source_folder in source_folders):
            existing_folders = filter(os.path.isdir, source_folders)
            existing_folders = list(existing_folders)

            for source_folder in existing_folders:
                print("Перенесем данные из '%s' в '%s'" % (source_folder, wavs_path))
                files = os.listdir(source_folder)
                for file in files:
                    source_file = os.path.join(source_folder, file)
                    destination_file = os.path.join(wavs_path, file)
                    shutil.move(source_file, destination_file)

                print("Удалим пустую '%s'" % source_folder)
                os.rmdir(source_folder)

    mels_path = os.path.join(dataset_path, 'mels')
    if not os.path.isdir(mels_path):
        os.mkdir(mels_path)
    mags_path = os.path.join(dataset_path, 'mags')
    if not os.path.isdir(mags_path):
        os.mkdir(mags_path)

    '''tqdm - визуальная обертка для более информативного отображения прогресса выполнения программы'''
    for fname in tqdm(speech_dataset.fnames):
        mel, mag = get_spectrograms(os.path.join(wavs_path, '%s.wav' % fname))

        t = mel.shape[0]
        # Marginal padding for reduction shape sync.
        num_paddings = hp.reduction_rate - (t % hp.reduction_rate) if t % hp.reduction_rate != 0 else 0
        mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
        mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
        # Reduction
        mel = mel[::hp.reduction_rate, :]

        np.save(os.path.join(mels_path, '%s.npy' % fname), mel)
        np.save(os.path.join(mags_path, '%s.npy' % fname), mag)
