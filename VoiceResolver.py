import librosa
import numpy
from keras.models import Sequential
from keras.models import load_model, save_model
import os
import operator
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam

import constants
from AudioManager import AudioManager


class VoiceResolver:
    
    def __init__(self):
        self.audio_manager = AudioManager()
        return

    def resolve_voice(self, audio_series):
        models_path = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(os.getcwd(), 'models')):
            models_path.extend(filenames)
            break
        predictions = {}
        for model_path in models_path:
            model = load_model(os.path.join(os.getcwd(), 'models', model_path))
            samples, _ = self.prepare_samples(audio_series, os.path.basename(model_path))
            prediction = model.predict(samples)
            predictions[model_path.split('.')[0]] = numpy.sum(prediction)
        
        print(predictions)
        return max(predictions.items(), key=operator.itemgetter(1))[0]

    def prepare_samples(self, audio_series, series_name, target=False):
        print(f"Preparing samples for {series_name}")

        data = librosa.stft(audio_series, n_fft=constants.MODEL_NFFT).swapaxes(0, 1)
        samples = []

        for i in range(0, len(data) - constants.BLOCK_LENGTH, constants.BLOCK_OVERLAP):
            samples.append(numpy.abs(data[i:i + constants.BLOCK_LENGTH]))

        results_shape = (len(samples), 1)
        results = numpy.ones(results_shape) if target else numpy.zeros(results_shape)
        return numpy.array(samples), results
    
    def get_training_samples(self):
        result = []
        dirname, languages, _ = next(os.walk(os.path.join(os.getcwd(), 'samples')))
        for language in languages:
            _, _, records = next(os.walk(os.path.join(dirname, language)))
            for record_path in records:
                result.append((
                    self.audio_manager.load_by_path(os.path.join(dirname, language, record_path)),
                    record_path.split('.')[0]
                    ))
        return result
        
        
    def train_for_new_voice(self, target_series, series_name):
        training_series = []
        print("Creating train data")
        for series in target_series:
            print(f"Adding {series_name} series to training set")
            training_series.append((series, series_name, True))
            
        training_series.extend(self.get_training_samples())
        
        samples = [
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
            ("russian", "male", 1, 2)
        ]
        
        X, Y = self.prepare_samples(*training_series[0])
        for series in training_series[1:]:
            dx, dy = self.prepare_samples(*series)
            X = numpy.concatenate((X, dx), axis=0)
            Y = numpy.concatenate((Y, dy), axis=0)
            del dx, dy
        
        perm = numpy.random.permutation(len(X))
        X = X[perm]
        Y = Y[perm]
        
        if os.path.exists(os.path.join(os.getcwd(), 'models', f'{series_name}')):
            model = load_model(os.path.join(os.getcwd(), 'models', f'{series_name}'))
        else:
            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=X.shape[1:]))
            model.add(LSTM(64))
            model.add(Dense(64))
            model.add(Activation('tanh'))
            model.add(Dense(16))
            model.add(Activation('sigmoid'))
            model.add(Dense(1))
            model.add(Activation('hard_sigmoid'))
        
        model.compile(
            Adam(lr=0.005), 
            loss='binary_crossentropy', 
            metrics=['accuracy'])
        print(f"Training model for {series_name}")
        model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2)
        save_model(model,
            os.path.join(os.getcwd(), "models", f"{series_name}.hd5"))
        

