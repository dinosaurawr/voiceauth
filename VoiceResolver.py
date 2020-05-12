import librosa
import numpy

import constants


class VoiceResolver:

    def resolve_voice(self, audio_series):
        return

    def prepare_samples(self, audio_series, series_name, is_training=False):
        print(f"Preparing samples for {series_name}")

        data = librosa.stft(audio_series, n_fft=constants.MODEL_NFFT).swapaxes(0, 1)
        samples = []

        for i in range(0, len(data) - constants.BLOCK_LENGTH, constants.BLOCK_OVERLAP):
            samples.append(numpy.abs(data[i:i + constants.BLOCK_LENGTH]))

        if is_training:
            results_shape = (len(samples), 1)
            results = numpy.ones(results_shape) if is_training else numpy.zeros(results_shape)
            return numpy.array(samples), results
        else:
            return numpy.array(samples)

    def train_for_new_voice(self, new_audio_series, series_name):
        print(f"Training new model for {series_name}")
        samples, expected_result = self.prepare_samples(new_audio_series, series_name, True)
        voices = [
            ("japanese", "female", 1, 1, True),
            ("japanese", "female", 1, 2, True),
            ("japanese", "female", 1, 3, True),
            ("french", "male", 1, 1, False),
            ("french", "male", 1, 2, False),
            ("english", "male", 1, 1, False)
        ]
        voice_series = []
        return

