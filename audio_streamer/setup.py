# setup.py
from setuptools import setup
import os
import sounddevice  # import before using

APP = ['audio_streamer.py']

# Locate PortAudio binaries shipped with sounddevice
portaudio_dir = os.path.join(os.path.dirname(sounddevice.__file__), '_sounddevice_data', 'portaudio-binaries')
portaudio_files = []
if os.path.exists(portaudio_dir):
    portaudio_files = [os.path.join(portaudio_dir, f) for f in os.listdir(portaudio_dir)]

OPTIONS = {
    'argv_emulation': True,
    'packages': ['numpy', 'sounddevice', 'setproctitle'],
    'resources': portaudio_files,  # include PortAudio dylib
    'plist': {
        'NSMicrophoneUsageDescription': 'Required to capture audio for streaming',
        'LSUIElement': True,  # run headless, no Dock icon
    },
}

setup(
    app=APP,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
