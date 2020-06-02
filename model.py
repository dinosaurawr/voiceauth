from ModelManager import ModelManager
from AudioManager import AudioManager

am = AudioManager()
sasha = am.load_last_record()
sasha_model = ModelManager(sasha, "Aleksandr")