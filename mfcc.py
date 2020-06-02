from AudioManager import AudioManager
from VoiceFeatures import VoiceFeatures

audiomanager = AudioManager()
audio = audiomanager.load_sample("japanese", "female", 1, 1)
features = VoiceFeatures(
    label='japaneseFemale1',
    save_coeffs=False,
    series=audio
)
print(features.mfcc)
print(features.resolve_voice())

