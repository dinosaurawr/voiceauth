import os

import numpy

import constants
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pyaudio
import wave


class AudioManager:

    def __init__(self):
        self.samples_directory = os.path.join(os.getcwd(), "samples")

    def load_sample(self, language, sex, person_number, track):
        file_path = os.path.join(
            self.samples_directory, 
            language, 
            f"{sex}{person_number}_{track}.wav"
        )
        series, _ = librosa.load(path=file_path, sr=constants.SAMPLE_RATE)
        return series

    def load_by_path(self, file_path):
        series, _ = librosa.load(path=file_path, sr=constants.SAMPLE_RATE)
        return series

    def record_sample(self):
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 2
        seconds = 3
        filename = "last_record.wav"

        p = pyaudio.PyAudio()

        print('Recording')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=constants.SAMPLE_RATE,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []

        for i in range(0, int(constants.SAMPLE_RATE / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        print('Finished recording, saving...')

        wf = wave.open(os.path.join(os.getcwd(), filename), 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(constants.SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return

    def load_last_record(self):
        file_path = os.path.join(
            os.getcwd(),
            "last_record.wav"
        )
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

    def get_samples_list(self):
        result = []
        recordings_dir = os.path.join(
            os.getcwd(),
            'free-spoken-digit-dataset',
            'recordings'
        )

        for filename in os.listdir(recordings_dir):
            if filename.endswith('.wav'):
                result.append(
                    os.path.join(recordings_dir, filename)
                )
        return result