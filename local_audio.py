#!/Users/israel/Development/tts_pipeline/mr_env/bin/python -u
import setproctitle
setproctitle.setproctitle("Hermes Listener")

import os
import select
import signal
import socket
import threading

from collections import deque

import numpy as np
import sounddevice as sd

# Audio parameters (match sender)
IP = "239.0.0.1"
PORT = 5005
CHANNELS = 1
RATE = 24000
FRAMES_PER_PACKET = 1024  # must match sender
SAMPLE_FORMAT = np.float32

# Queue Parameters
MAX_QUEUE_PACKETS = None  # max packets to buffer (drop old if full)
START_THRESHOLD = 3 # minimum packets to start playback

def _sigterm_handler(signum, frame):
    exit_event.set()
    
signal.signal(signal.SIGTERM, _sigterm_handler)

# Rolling buffer
audio_queue = deque(maxlen=MAX_QUEUE_PACKETS)

# Setup UDP multicast
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
sock.bind(('', PORT))
mreq = socket.inet_aton(IP) + socket.inet_aton('0.0.0.0')
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
sock.setblocking(False)

playing = threading.Event()
exit_event = threading.Event()

played_packets = 0
SILENCE = np.zeros((FRAMES_PER_PACKET, CHANNELS), dtype=SAMPLE_FORMAT)
def callback(outdata, frames, time, status):
    global played_packets

    if not playing.is_set():
        played_packets = 0
        if len(audio_queue) >= START_THRESHOLD:
            playing.set()  # start playback once we have enough buffered
        else:
            outdata[:] = SILENCE
            return
        
    if audio_queue:
        audio = audio_queue.popleft()
        played_packets += 1
        if audio.shape[0] < frames:
            padding = np.zeros((frames - audio.shape[0], CHANNELS), dtype=SAMPLE_FORMAT)
            audio = np.vstack((audio, padding))
        outdata[:] = audio
    else:
        # no data available: output silence
        outdata[:] = SILENCE
        playing.clear()  # stop playback until we have more data
        print(f"Played {played_packets} packets")

def recv_thread():
    while not exit_event.is_set():
        select.select([sock], [], [])
        while True:
            try:
                data, _ = sock.recvfrom(CHANNELS * FRAMES_PER_PACKET * np.dtype(SAMPLE_FORMAT).itemsize)
                audio = np.frombuffer(data, dtype=SAMPLE_FORMAT).reshape(-1, CHANNELS)
                audio_queue.append(audio)
            except BlockingIOError:
                break
        
   
recv_loop = threading.Thread(target=recv_thread, daemon=True)
recv_loop.start()

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
    try:
        exit_event.wait()  # Blocks forever, zero CPU
    except KeyboardInterrupt:
        exit_event.set()
    
    recv_loop.join(timeout=2)
        
    print("Exited Hermes Listener")