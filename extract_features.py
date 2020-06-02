from python_speech_features import mfcc, delta
from sklearn.preprocessing import scale
from numpy import hstack

def extract_features(series, rate):
    mfcc_feature = mfcc(series, rate)
    mfcc_feature = scale(mfcc)
    delta_feature = delta(mfcc, 1)
    combined = hstack((mfcc_feature, delta_feature))
    return combined