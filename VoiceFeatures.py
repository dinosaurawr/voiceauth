import librosa
import numpy
import os
import constants
import operator

class VoiceFeatures:

    def __init__(self, label='Unlabled', load_from_file=False, save_coeffs=False, series=[]):

        self.label = label
        self.coeffs_folder = os.path.join(os.getcwd(), 'mfcc_coeffs')
        self.series = series
        if (load_from_file):
            self.mfcc = self.load_coefs_from_file_by_label()
        else:
            self.mfcc = self.compute_mfcc()
            if (save_coeffs):
                self.save_coefs()

        
    def compute_mfcc(self):
        mfcc = librosa.feature.mfcc(
            self.series,
            sr=constants.SAMPLE_RATE,
            n_mfcc=constants.MFCC_COUNT,
            n_fft=2048
        )
        
        mfcc = numpy.sum(mfcc[2:], axis=-1)
        mfcc = mfcc / numpy.max(numpy.abs(mfcc))
        return mfcc

    def get_euclidian_distance_with_other(self, mfcc_vector):
        return numpy.sum((self.mfcc - mfcc_vector) ** 2)

    def save_coefs(self):
        file = open(
            os.path.join(
                self.coeffs_folder,
                f'{self.label}.txt'
            ),
            'w'
        )
        file.write(
            ' '.join([str(coeff) for coeff in self.mfcc])
        )
        file.close()
        return

    def load_coefs_from_file_by_label(self):
        file = open(
            os.path.join(
                self.coeffs_folder,
                f'{self.label}.txt'
            ),
            'r'
        )
        cont = file.read()
        self.mfcc = numpy.array(cont.split(' '), numpy.float32)
        file.close()
        return self.mfcc

    def resolve_voice(self):
        print(self.mfcc)
        print(len(self.mfcc))
        if len(self.mfcc) == 0:
            return 'Not enough MFCC coeffs'
        coefs_files = [
            file for file in os.listdir(self.coeffs_folder)
            if file.endswith('.txt')
        ]
        results = {}
        for filename in coefs_files:
            other_mfcc = numpy.array(
                open(os.path.join(self.coeffs_folder, filename), 'r').read().split(' '),
                dtype=numpy.float32)
            euclidian = self.get_euclidian_distance_with_other(other_mfcc)
            same_by = round(1 /euclidian * 100, 2)
            print(f'{self.label} same with {filename.split(".")[0]} is {same_by}')
            results[filename] = same_by
        return max(results.items(), key=operator.itemgetter(1))[0]
        