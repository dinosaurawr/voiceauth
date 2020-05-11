import python_speech_features as speech_features
import numpy

import constants


class VoiceFeatures:

    def __init__(self, audio_series, sample_rate):
        self.sample_rate = sample_rate
        self.audio_series = audio_series
        self.mfcc_feature_vector = []
        self.filter_bank = []
        self.ssc = []

    def compute_mfcc_feature_vector(self):
        mfcc = speech_features.mfcc(
            signal=self.audio_series,
            samplerate=self.sample_rate,
            winlen=constants.MFCC_WIN_LEN,
            numcep=constants.MFCC_COUNT,
            nfft=constants.NFFT
        )
        mfcc_sum = numpy.sum(mfcc, axis=0)
        self.mfcc_feature_vector = mfcc_sum / numpy.max(numpy.abs(mfcc_sum))

    def compute_filter_bank(self):
        self.filter_bank, _ = speech_features.fbank(
            signal=self.audio_series,
            samplerate=self.sample_rate
        )

    def compute_ssc(self):
        self.ssc = speech_features.ssc(
            signal=self.audio_series,
            samplerate=self.sample_rate
        )
