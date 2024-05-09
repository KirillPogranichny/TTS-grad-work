import os
import copy
import librosa
import scipy.io.wavfile as wavfile
import numpy as np
import shutil

from tqdm import tqdm
from scipy import signal
from hparams import HParams as hp
from datasets.ru_speech import RUSpeech


def spectrogram2wav(mag):
    ''' Генерация wav файла из спектрограммы
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # Транспонирование
    mag = mag.T

    # Денормализация
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag ** hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # Обрезка
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Применяет алгоритм Грифиина-Лима для восстановления сигнала из спектрограммы
    Args:
      spectrogram (numpy.ndarray): Спектрограммма, из которой нужно восстановить сигнал.
    Return:
      y (numpy.ndarray): Восстановленный сигнал.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
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
    return librosa.istft(spectrogram, hop_length=hp.hop_length, win_length=hp.win_length, window="hann")


def get_spectrograms(fpath):
    '''Функция для получения спектрограмм и мел-спектрограмм из wav файлов.
    Args:
      fpath(str): The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.  # Мел-спектрограмма
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.  # Спектрограмма
    '''
    # Загрузка аудиофайла
    try:
        y, sr = librosa.load(fpath, sr=hp.sr)
    except Exception as e:
        print(f'Ошибка при загрузке файла {fpath}: {e}')
        return None, None

    # Обрезка
    y, _ = librosa.effects.trim(y)

    # Преэмфазирование
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # Кратковременное преобразование Фурье
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # Спектрограмма
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # Мел-спектрограмма
    mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # Преобразование в децибелы
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Нормализация
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Транспонирование
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def save_to_wav(mag, filename):
    """Generate and save an audio file from the given linear spectrogram using Griffin-Lim."""
    wav = spectrogram2wav(mag)

    try:
        wavfile.write(filename, hp.sr, wav)
        print(f"Аудиофайл '{filename}' успешно сохранен.")
    except Exception as e:
        print(f"Ошибка при сохранении аудиофайла '{filename}': {e}")


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
                print(f"Перенесем данные из '{source_folder}' в '{wavs_path}'")
                files = os.listdir(source_folder)
                for file in files:
                    source_file = os.path.join(source_folder, file)
                    destination_file = os.path.join(wavs_path, file)
                    shutil.move(source_file, destination_file)

                print(f"Удалим пустую папку '{source_folder}'")
                os.rmdir(source_folder)

    mels_path = os.path.join(dataset_path, 'mels')
    if not os.path.isdir(mels_path):
        os.mkdir(mels_path)
    mags_path = os.path.join(dataset_path, 'mags')
    if not os.path.isdir(mags_path):
        os.mkdir(mags_path)

    for fname in tqdm(speech_dataset.fnames):
        mel, mag = get_spectrograms(os.path.join(wavs_path, f'{fname}.wav'))

        t = mel.shape[0]
        # Marginal padding for reduction shape sync.
        num_paddings = hp.reduction_rate - (t % hp.reduction_rate) if t % hp.reduction_rate != 0 else 0
        mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
        mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
        # Reduction
        mel = mel[::hp.reduction_rate, :]

        np.save(os.path.join(mels_path, f'{fname}.npy'), mel)
        np.save(os.path.join(mags_path, f'{fname}.npy'), mag)
