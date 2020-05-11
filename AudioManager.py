import os

import numpy

import constants
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioManager:

    def __init__(self):
        self.samples_directory = os.path.join(os.getcwd(), "samples")

    def load_sample(self, language, sex, person_number, track):
        file_path = os.path.join(self.samples_directory, language, f"{sex}{person_number}_{track}.wav")
        series, _ = librosa.load(path=file_path, sr=constants.SAMPLE_RATE)
        return self.filter(audio_series=series)

    def filter(self, audio_series):
        apower = librosa.amplitude_to_db(
            numpy.abs(librosa.stft(audio_series, n_fft=constants.NFFT)), ref=numpy.max
        )

        apsums = numpy.sum(apower, axis=0) ** 2
        apsums -= numpy.min(apsums)
        apsums /= numpy.max(apsums)

        apsums = numpy.convolve(apsums, numpy.ones((9,)), 'same')

        apsums -= numpy.min(apsums)
        apsums /= numpy.max(apsums)

        apsums = numpy.array(apsums > 0.35, dtype=bool)

        apsums = numpy.repeat(
            apsums,
            numpy.ceil(len(audio_series) / len(apsums)))[:len(audio_series)]

        return audio_series[apsums]
