import os
import tarfile


def prepare_dataset():
    root_path = 'C:\\Users\\dinos\\.keras\\datasets\\archive'
    for filename in os.listdir(root_path):
        print('Preparing file:', filename)
        tar = tarfile.open(os.path.join(root_path, filename))
        tar.extractall(os.path.join('dataset', filename))
        tar.close()


prepare_dataset()
