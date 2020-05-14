from keras import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam
import numpy
import audio_manager
import constants

def create_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=[constants.MFCC_COUNT]))
    model.add(LSTM(64))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(16))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('hard_sigmoid'))

    model.compile(Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
def train_model_for_last_three_records(vectors):
    model = create_model()
    model.fit(x=[vectors], y=numpy.ones(len(vectors)), batch_size=1)