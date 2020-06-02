from sklearn.mixture import GaussianMixture
from os import path, getcwd, listdir
from MFCCFeatures import MFCCFeatures
from AudioManager import AudioManager
import numpy

class ModelManager():

    def __init__(self, target_series, sr):
        self.features = MFCCFeatures(target_series, sample_rate=sr)
        self.am = AudioManager()
        self.train_data_dir = path.join(
            getcwd(),
            'free-spoken-digit-dataset'
        )
        self.sample_tuples = self.__get_indexed_sample_files()
        return


    def __get_indexed_sample_files(self):
        result = []
        for filename in listdir(self.train_data_dir):
            if (filename.endswith('.wav')):
                full_path = path.join(self.train_data_dir, filename)
                splitted = filename.split('.')[0].split('_')
                result.append(
                    (full_path, splitted[1], splitted[2])
                )
        return result

    def __get_train_features(self, filepath):
        randoms = numpy.random.randint(0, len(self.sample_tuples) - 1, 20)
        chosed_samples = self.sample_tuples[randoms]
        for sample in chosed_samples:
