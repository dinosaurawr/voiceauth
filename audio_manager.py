from librosa import load
from os import path, getcwd
import constants
from pyaudio import PyAudio, paInt16
from wave import open
import time


SAMPLES_FOLDER = path.join(getcwd(), 'samples')

RECORD_CHUNK = 1024
RECORD_SAMPLE_FORMAT = paInt16
RECORD_CHANNELS = 1
RECORD_SECONDS = 3
RECORD_FILENAME = 'last_record.wav'

def get_sample_data(language, sex, person, track):
    data, sample_rate = load(
        path=path.join(
            SAMPLES_FOLDER,
            language,
            f'{sex}{person}_{track}'
            ),
        sr=constants.SAMPLE_RATE
        )
    return data

def get_data_by_path(path):
    data, sample_rate = load(
        path=path,
        sr=constants.SAMPLE_RATE
        )
    return data

def record_and_get_data(filename_to_save_record):
    p = PyAudio()
    print('Recording')

    stream = p.open(format=RECORD_SAMPLE_FORMAT,
                    channels=RECORD_CHANNELS,
                    rate=constants.SAMPLE_RATE,
                    frames_per_buffer=RECORD_CHUNK,
                    input=True)

    frames = []

    for i in range(0, int(constants.SAMPLE_RATE / RECORD_CHUNK * RECORD_SECONDS)):
        data = stream.read(RECORD_CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print('Finished recording, saving...')
    
    wf = open(path.join(getcwd(), filename_to_save_record), 'wb')
    wf.setnchannels(RECORD_CHANNELS)
    wf.setsampwidth(p.get_sample_size(RECORD_SAMPLE_FORMAT))
    wf.setframerate(constants.SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    data, _ = load(
        path.join(getcwd(), filename_to_save_record), 
        sr=constants.SAMPLE_RATE,
        mono=True)
    return data

def record_three_samples_and_get_series():
    print('Запишите фразу с помощью которой будете проводить аутентификацию')
    time.sleep(3)
    result = []
    for i in range(2):
        series = record_and_get_data(f'record{i}.wav')
        time.sleep(2)
        result.append(series)
    return result