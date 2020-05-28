from tensorflow.keras.models import load_model
import numpy
from AudioManager import AudioManager
import os
import librosa
import constants

audio_manager = AudioManager()
model = load_model('model.hdf5')

series = audio_manager.load_by_path(os.path.join(os.getcwd(), 'record.wav'))
#series = audio_manager.load_sample(language="russian",sex="male",person_number=1,track=2)

data = librosa.stft(series, n_fft=constants.MODEL_NFFT).swapaxes(0, 1)
samples = []

for i in range(0, len(data) - constants.BLOCK_LENGTH, constants.BLOCK_OVERLAP):
    samples.append(numpy.abs(data[i:i + constants.BLOCK_LENGTH]))

samples = numpy.array(samples)
print(samples.shape)

result = model.predict(samples)
print(result)