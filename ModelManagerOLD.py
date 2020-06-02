from keras import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
import librosa
import numpy
import constants
from AudioManager import AudioManager

class ModelManager():
    
    def __init__(self, target_series, target_series_name,  model_filename=""):
        self.fit_new = len(model_filename) == 0
        self.audio_manager = AudioManager()
        
        self.samples = [
            ("japanese", "female", 1, 1),
            ("japanese", "female", 1, 2),
            ("japanese", "female", 1, 3),
            ("japanese", "female", 1, 4),
            ("french", "male", 1, 1),
            ("french", "male", 1, 2),
            ("french", "female", 1, 1),
            ("french", "female", 1, 2),
            ("french", "female", 1, 3),
            ("english", "male", 1, 1),
            ("english", "male", 2, 1),
            ("english", "male", 2, 2),
            ("english", "male", 2, 3),
            ("russian", "male", 1, 1),
            ("russian", "male", 1, 2),    
        ]

        if not self.fit_new:
            self.model = load_model(
                os.path.join(
                    os.getcwd(), 
                    'models', 
                    model_filename
                )
            )
        else:
            self.model = Sequential()
            self.fit_from_sample_data(target_series, target_series_name)
            
    def create_model(self, input_shape):
        self.model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(64))
        self.model.add(Dense(64))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(16))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(1))
        self.model.add(Activation('hard_sigmoid'))
        self.model.compile(Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])
        return
    
    def get_prediction_result(self, prediction_series):
        data = librosa.stft(prediction_series, n_fft=constants.MODEL_NFFT).swapaxes(0, 1)
        samples = []
    
        for i in range(0, len(data) - constants.BLOCK_LENGTH, constants.BLOCK_OVERLAP):
            samples.append(numpy.abs(data[i:i + constants.BLOCK_LENGTH]))
            
        result = self.model.predict(numpy.array(samples))
        return result
    
    def fit_from_sample_data(self, target_series, target_series_name):
        X, Y = self.prepare_target_data()
        for sample in self.samples:
            dx, dy = self.prepare_samples_data(*sample)
            X = numpy.concatenate((X, dx), axis=0)
            Y = numpy.concatenate((Y, dy), axis=0)
            del dx, dy
            
        perm = numpy.random.permutation(len(X))
        X = X[perm]
        Y = Y[perm]
        
        self.create_model(X.shape[1:])
        self.model.fit(X, Y, epochs=15, batch_size=34, validation_split=0.2)
        print(self.model.evaluate(X, Y))
        self.model.save(
            os.path.join(
                os.getcwd(),
                'models',
                f'{target_series_name}.hdf5'
            )    
        )        
        
    def prepare_target_data(self):
        series = self.audio_manager.load_last_record()
        
        target_data = librosa.stft(series, n_fft=constants.MODEL_NFFT).swapaxes(0, 1)
        samples = []
    
        for i in range(0, len(target_data) - constants.BLOCK_LENGTH, constants.BLOCK_OVERLAP):
            samples.append(numpy.abs(target_data[i:i + constants.BLOCK_LENGTH]))
            
        return numpy.array(samples), numpy.ones((len(samples), 1))
        
    def prepare_samples_data(self, language, sex, person_number, track):
        print(f"Loadig sample of {language} {sex}, person: {person_number}, track: {track}")
        
        series = self.audio_manager.load_sample(
            language=language,
            sex=sex,
            person_number=person_number,
            track=track
        )
            
        data = librosa.stft(series, n_fft=constants.MODEL_NFFT).swapaxes(0, 1)
        samples = []
    
        for i in range(0, len(data) - constants.BLOCK_LENGTH, constants.BLOCK_OVERLAP):
            samples.append(numpy.abs(data[i:i + constants.BLOCK_LENGTH]))
 
        return numpy.array(samples), numpy.zeros((len(samples), 1))
        