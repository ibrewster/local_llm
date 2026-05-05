#!/Users/israel/Development/tts_pipeline/llm_env/bin/python -u
import setproctitle
setproctitle.setproctitle("Hermes Player")

import signal
import threading
import urllib
import urllib.request
import urllib.error

from urllib.parse import quote

import multiprocessing as mp
import multiprocessing.shared_memory as shm
import sounddevice as sd

import numpy as np

from receiver_config import AIRPLAY_ID

# Audio parameters (match sender)
IP = "239.0.0.1"
PORT = 5005
CHANNELS = 1
RATE = 24000
FRAMES_PER_PACKET = 1024  # must match sender
SAMPLE_FORMAT = np.float32

# Queue Parameters
START_THRESHOLD = 3 # minimum packets to start playback

# Pre-allocate a ring buffer of shared memory blocks
RING_SIZE = 4096  # number of slots
MASK = RING_SIZE - 1
BLOCK_BYTES = CHANNELS * FRAMES_PER_PACKET * np.dtype(np.float32).itemsize


class BidirectionalEvent:
    """
    A drop-in replacement for threading.Event where wait() means
    "wait for a state change" rather than "wait for set".

    - If the event is currently clear, wait() blocks until it is set.
    - If the event is currently set,  wait() blocks until it is cleared.

    set/clear/is_set behave identically to threading.Event.
    """

    def __init__(self):
        self._set_event   = threading.Event()
        self._clear_event = threading.Event()
        self._clear_event.set()

    def set(self):
        self._clear_event.clear()
        self._set_event.set()

    def clear(self):
        self._set_event.clear()
        self._clear_event.set()

    def is_set(self) -> bool:
        return self._set_event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """
        Block until the state changes from what it is at call time.
        Returns True if the change occurred, False on timeout.
        """
        if self._set_event.is_set():
            return self._clear_event.wait(timeout=timeout)
        else:
            return self._set_event.wait(timeout=timeout)
        
    def wait_set(self, timeout: float | None = None) -> bool:
        return self._set_event.wait(timeout=timeout)
    
    def wait_clear(self, timeout: float | None = None) -> bool:
        return self._clear_event.wait(timeout=timeout)


def receiver_process(shm_name, read_idx, write_idx, exit_flag):
    """Runs in a separate process - owns the socket, no GIL contention with audio"""
    import socket, numpy as np

    def _sigterm_handler(signum, frame):
        print("Signal handler in RECEIVER called")
        raise KeyboardInterrupt("SIGTERM received from launchd in RECEIVER")
    
    signal.signal(signal.SIGTERM, _sigterm_handler)    

    setproctitle.setproctitle("Hermes Receiver")
    try:
        existing_shm = shm.SharedMemory(name=shm_name, track=False)
    except TypeError:
        # Python < 3.13
        existing_shm = shm.SharedMemory(name=shm_name)
    
    ring = np.ndarray((RING_SIZE, FRAMES_PER_PACKET, CHANNELS), 
                      dtype=np.float32, buffer=existing_shm.buf)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(3.5 * 1024 * 1024))
    actual_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"Allocated SO_RCVBUF: {actual_size} bytes")    
    sock.bind(('', PORT))
    mreq = socket.inet_aton(IP) + socket.inet_aton('0.0.0.0')
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(0.5)
    
    ring_views = [memoryview(ring[i]) for i in range(RING_SIZE)]

    try:
        while not exit_flag.is_set():
            try:
                slot = write_idx.value & MASK
                sock.recv_into(ring_views[slot], BLOCK_BYTES)
                
                write_idx.value += 1
                if (write_idx.value - read_idx.value) > RING_SIZE:
                    read_idx.value = write_idx.value - RING_SIZE

            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("Received Keyboard Inerrupt. Exiting.")
        exit_flag.set()
        
    print("Exiting Receiver Process")
    existing_shm.close()
    sock.close()
    print("Receiver process complete")


def set_airplay_volume(level):
    """
    Sends a PUT request using the Python standard library.
    level: integer (0-100)
    """
    url = f"http://10.27.81.2:8181/airplay_devices/{quote(AIRPLAY_ID, safe='')}/volume"
    AIRPLAY_ID
    payload_bytes = urllib.parse.urlencode({"level": level}).encode('utf-8')
    
    req = urllib.request.Request(url, data=payload_bytes, method='PUT')
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    
    try:
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status in (200, 204)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"Volume ducking failed: {e}")
        return False
    
        
def audio_ducker(duck_event: BidirectionalEvent):
    if AIRPLAY_ID is None:
        print("Not running ducker on this node")
        return # No device to duck.
    
    print("Ducker Started")
    try:
        while not exit_flag.is_set():
            if not duck_event.wait_set(timeout=0.5):
                continue
            print("Ducking Airplay Volume")
            set_airplay_volume(50)
            duck_event.wait_clear()
            print("Restoring Airplay Volume")
            set_airplay_volume(100)
    except KeyboardInterrupt:
        pass
    
    print("Ducker thread exited")
    
# Main process
if __name__ == '__main__':
    exit_flag = mp.Event()
    
    def _sigterm_handler(signum, frame):
        print("Signal handler in MAIN called")
        raise KeyboardInterrupt("SIGTERM received from launchd IN MAIN")
    
    signal.signal(signal.SIGTERM, _sigterm_handler)
    
    write_idx = mp.Value('Q', 0, lock=False)
    read_idx = mp.Value('Q', 0, lock=False)  
    
    ring_shm = shm.SharedMemory(create=True, size=RING_SIZE * BLOCK_BYTES)
    ring = np.ndarray((RING_SIZE, FRAMES_PER_PACKET, CHANNELS), 
                      dtype=np.float32, buffer=ring_shm.buf)

    proc = mp.Process(
        target=receiver_process,
        args=(ring_shm.name, read_idx, write_idx, exit_flag),
        daemon=True
    )
    proc.start()

    playing = False
    duck_audio = BidirectionalEvent()
    
    volume_thread = threading.Thread(target=audio_ducker, daemon=True, args=(duck_audio, ))
    volume_thread.start()
    
    played_packets = 0
    SILENCE = np.zeros((FRAMES_PER_PACKET, CHANNELS), dtype=np.float32)
    def callback(outdata, frames, time, status):
        global played_packets, playing

        available = write_idx.value - read_idx.value
        if not playing:
            played_packets = 0
            if available >= START_THRESHOLD:
                playing = True
                duck_audio.set()
            else:
                if available == 1:
                    duck_audio.set()
                outdata[:] = SILENCE
                return

        if available > 0:
            slot = read_idx.value & MASK
            outdata[:] = ring[slot]
            read_idx.value += 1
            played_packets += 1
        else:
            outdata[:] = SILENCE
            playing = False
            duck_audio.clear()
            print(f"Played {played_packets} packets")


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
            exit_flag.wait()  # Blocks forever, zero CPU
            print("exit_flag is now set")
        except KeyboardInterrupt:
            print("Received keyboard interrupt. Exiting")
        finally:
            print("Setting exit flag to signal child")
            exit_flag.set()
    
    print("Hermes Streamer Exiting.")
    proc.join(timeout=2)
    if proc.is_alive():
        print("!!Killing child process after two seconds")
        proc.kill()
        
    if volume_thread.is_alive():
        duck_audio.clear()
        
    ring_shm.close()
    ring_shm.unlink()

    print("Exited Hermes Listener")