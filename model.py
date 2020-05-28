import librosa
import numpy
from keras import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam

import constants
import os
from AudioManager import AudioManager


def prepare_audio(language, sex, person_number, track, recorded=False):
    print(f"Loadig sample of {language} {sex}, person: {person_number}, track: {track}")
    audio_manager = AudioManager()
    
    if recorded:
        series = audio_manager.load_by_path(os.path.join(os.getcwd(), 'record.wav'))
    else:
        series = audio_manager.load_sample(
            language=language,
            sex=sex,
            person_number=person_number,
            track=track
        )
        
    data = librosa.stft(series, n_fft=constants.MODEL_NFFT).swapaxes(0, 1)
    samples = []

    for i in range(0, len(data) - constants.BLOCK_LENGTH, constants.BLOCK_OVERLAP):
        samples.append(numpy.abs(data[i:i + constants.BLOCK_LENGTH]))

    results_shape = (len(samples), 1)
    results = numpy.ones(results_shape) if recorded else numpy.zeros(results_shape)
    return numpy.array(samples), results


sample_voices = [
    ("japanese", "female", 1, 1, False),
    ("japanese", "female", 1, 2, False),
    ("japanese", "female", 1, 3, False),
    ("japanese", "female", 1, 4, False),
    ("french", "male", 1, 1, False),
    ("french", "male", 1, 2, False),
    ("french", "female", 1, 1, False),
    ("french", "female", 1, 2, False),
    ("french", "female", 1, 3, False),
    ("english", "male", 1, 1, False),
    ("english", "male", 2, 1, False),
    ("english", "male", 2, 2, False),
    ("english", "male", 2, 3, False),
    ("russian", "male", 1, 1, False),
    ("russian", "male", 1, 2, False),
]

sample_voices.insert(0, ("", "", 1, 1, True))

X, Y = prepare_audio(*sample_voices[0])
for voice in sample_voices[1:]:
    dx, dy = prepare_audio(*voice)
    X = numpy.concatenate((X, dx), axis=0)
    Y = numpy.concatenate((Y, dy), axis=0)
    del dx, dy

perm = numpy.random.permutation(len(X))
X = X[perm]
Y = Y[perm]

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=X.shape[1:]))
model.add(LSTM(64))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dense(16))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))

model.compile(Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=15, batch_size=34, validation_split=0.2)

print(X.shape)
print(Y.shape)
print(model.evaluate(X, Y))
model.save("model.hdf5")