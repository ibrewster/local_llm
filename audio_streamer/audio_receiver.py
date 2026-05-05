import socket
import threading

from collections import deque

import numpy as np
import sounddevice as sd


# Audio parameters (match sender)
IP = "239.0.0.1"
PORT = 5004
CHANNELS = 2
RATE = 48000
FRAMES_PER_PACKET = 2048  # must match sender
SAMPLE_FORMAT = np.int16

# Queue Parameters
MAX_QUEUE_PACKETS = 100  # max packets to buffer (drop old if full)
START_THRESHOLD = 3 # minimum packets to start playback

# Rolling buffer
audio_queue = deque(maxlen=MAX_QUEUE_PACKETS)

# Setup UDP multicast
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', PORT))
mreq = socket.inet_aton(IP) + socket.inet_aton('0.0.0.0')
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
sock.setblocking(False)

playing = threading.Event()

def recv_loop():
    """Read all available UDP packets into the deque."""
    while True:
        try:
            data, _ = sock.recvfrom(CHANNELS * FRAMES_PER_PACKET * 2)
            audio_queue.append(data)  # old packets dropped automatically if full
        except BlockingIOError:
            break  # no more packets

def callback(outdata, frames, time, status):
    recv_loop()

    if not playing.is_set():
        if len(audio_queue) >= START_THRESHOLD:
            playing.set()  # start playback once we have enough buffered
        else:
            outdata[:] = np.zeros((frames, CHANNELS), dtype=SAMPLE_FORMAT)
            return
        
    if audio_queue:
        packet = audio_queue.popleft()
        audio = np.frombuffer(packet, dtype=SAMPLE_FORMAT).reshape(-1, CHANNELS)
        if audio.shape[0] < frames:
            padding = np.zeros((frames - audio.shape[0], CHANNELS), dtype=SAMPLE_FORMAT)
            audio = np.vstack((audio, padding))
        outdata[:] = audio
    else:
        # no data available: output silence
        outdata[:] = np.zeros((frames, CHANNELS), dtype=SAMPLE_FORMAT)
        playing.clear()  # stop playback until we have more data

# Sounddevice output stream
with sd.OutputStream(
    samplerate=RATE,
    blocksize=FRAMES_PER_PACKET,
    channels=CHANNELS,
    dtype=SAMPLE_FORMAT,
    callback=callback,
    latency='low'
):
    print("Listening for UDP audio...")
    while True:
        sd.sleep(1000)