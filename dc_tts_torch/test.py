import librosa
import soundfile
from hparams import HParams as hp
import csv


# def main():
#     filename = 'samples/en/1-wav.wav'
#     y, _ = librosa.load(filename)
#     steps = float(input("Number of semitones to shift audio file: "))
#     new_y = librosa.effects.pitch_shift(y, sr=hp.sr, n_steps=steps)
#     new_y = librosa.effects.time_stretch(new_y, rate=0.9)
#     soundfile.write("samples/en/pitchShifted.wav", new_y, hp.sr)
#
#
# main()

