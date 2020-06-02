from sklearn.mixture import GaussianMixture
from extract_features import extract_features
from os import path, getcwd, listdir
from numpy.random import random_integers
from numpy import asarray, vstack
from librosa import load
import pickle

SAMPLING_RATE = 8000

samples_folder = path.join(
    getcwd(),
    'free-spoken-digit-dataset'
)

models_folder = path.join(
    getcwd(),
    'models'
)

samples_paths = [path for path in listdir(samples_folder) if path.endswith('.wav')]

models_paths = [model_path for model_path in listdir(models_folder) if model_path.endswith('.model')]

def get_random_train_data(audio_count):

    integers = random_integers(0, len(samples_paths) + 1, audio_count)
    chosed_samples_paths = samples_paths[integers]

    result = asarray(())
    for sample_path in chosed_samples_paths:
        print(f'Exctracting features for {sample_path}')
        audio, _ = load(sample_path, SAMPLING_RATE)
        features = extract_features(audio, SAMPLING_RATE)
        
        if result.size == 0:
            result = features
        else:
            result = vstack((result, features))

    return result

def fit_speaker_model(features, label):
    gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
    gmm.fit()

    file = open(
        path.join(
            models_folder,
            f'{label}.gmm'
        ),
        'x'
    )

    pickle.dump(gmm, file)
    