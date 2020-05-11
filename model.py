import librosa
import numpy
from keras import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam

import constants
from AudioManager import AudioManager


def prepare_audio(language, sex, person_number, track, target=False):
    print(f"Loadig sample of {language} {sex}, person: {person_number}, track: {track}")
    audio_manager = AudioManager()
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
    results = numpy.ones(results_shape) if target else numpy.zeros(results_shape)
    return numpy.array(samples), results


voices = [
    ("japanese", "female", 1, 1, True),
    ("japanese", "female", 1, 2, True),
    ("japanese", "female", 1, 3, True),
    ("french", "male", 1, 1, False),
    ("french", "male", 1, 2, False),
    ("english", "male", 1, 1, False)
]

X, Y = prepare_audio(*voices[0])
for voice in voices[1:]:
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
model.fit(X, Y, epochs=15, batch_size=32, validation_split=0.2)

print(model.evaluate(X, Y))
model.save("model.hdf5")