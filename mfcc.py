import librosa
import numpy
import os
import VoiceFeatures
import constants
from AudioManager import AudioManager


def euclidean_distance(vector1, vector2):
    return numpy.sqrt(numpy.sum((vector1 - vector2) ** 2))


audiomanager = AudioManager()
audio = audiomanager.load_sample("japanese", "female", 1, 1)
audio2 = audiomanager.load_sample("japanese", "female", 1, 2)
features = VoiceFeatures.VoiceFeatures(audio_series=audio, sample_rate=constants.SAMPLE_RATE)
features2 = VoiceFeatures.VoiceFeatures(audio_series=audio2, sample_rate=constants.SAMPLE_RATE)
features2.compute_mfcc_feature_vector()
features.compute_mfcc_feature_vector()
print(features.mfcc_feature_vector)
print(euclidean_distance(features.mfcc_feature_vector, features2.mfcc_feature_vector))
print("\n")
