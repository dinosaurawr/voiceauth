from python_speech_features import mfcc
from numpy import sum, sqrt
import constants
import audio_manager

def get_mfcc_feature_vector(audio_series):
    mfcc_by_all_freq = mfcc(
        signal=audio_series,
        winlen=constants.MFCC_WIN_LEN,
        samplerate=constants.SAMPLE_RATE,
        numcep=constants.MFCC_COUNT
        )
    freq_sum = sum(mfcc_by_all_freq, axis=0)
    normalized = freq_sum / sqrt(sum(freq_sum ** 2))
    return normalized

def get_last_records_mfcc_vectors():
    result = []
    result_vectors = audio_manager.record_three_samples_and_get_series()
    for vector in result_vectors:
        result.append(get_mfcc_feature_vector(vector))
    return result