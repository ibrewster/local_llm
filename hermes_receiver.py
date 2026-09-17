#!/Users/israel/Development/tts_pipeline/llm_env/bin/python -u
"""Receive multicast audio from the Hermes sender and play it locally.

This script listens for UDP audio packets on a multicast address, stores them in a
shared ring buffer so the receiver process can decouple socket I/O from the audio
callback thread, and streams the buffered PCM samples to the local speaker. It also
optionally ducks AirPlay volume while playback is active so the local speaker output
remains audible without overwhelming the AirPlay stream.
"""

import setproctitle
setproctitle.setproctitle("Hermes Stream Player")

import signal
import threading
import time
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
UTTERANCE_GAP_SECONDS = 1.0  # silence required before the next utterance gets a chime

# Pre-allocate a ring buffer of shared memory blocks
RING_SIZE = 4096  # number of slots
MASK = RING_SIZE - 1
BLOCK_BYTES = CHANNELS * FRAMES_PER_PACKET * np.dtype(np.float32).itemsize


def make_start_chime() -> np.ndarray:
    """Create a prominent but pleasant chime that cuts through background music."""
    # Slightly lengthened the notes to give the ducker more time to engage
    notes = ((440.0, 0.4), (660.0, 0.9))
    parts = []

    for frequency, duration in notes:
        samples = max(1, int(RATE * duration))
        t = np.arange(samples, dtype=np.float32) / RATE

        # Added a 3rd harmonic (3x frequency) at 10% volume.
        # This acts like an acoustic "highlighter" to help it pierce through a dense music mix.
        note = (
                np.sin(2 * np.pi * frequency * t) +
                0.20 * np.sin(2 * np.pi * frequency * 2 * t) +
                0.10 * np.sin(2 * np.pi * frequency * 3 * t)
        )

        # Slower exponential decay (-2.0 instead of -4.0).
        # This keeps the volume higher for longer before fading out.
        envelope = np.exp(-2.0 * t)

        attack_samples = min(int(RATE * 0.05), samples // 2)
        if attack_samples > 0:
            envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples, endpoint=False)

        # Boosted overall volume from 0.25 to 0.70.
        # The math stays safely below clipping (1.0 + 0.2 + 0.1 = 1.3 max * 0.7 = 0.91)
        parts.append(note * envelope * 0.70)

    chime = np.concatenate(parts)

    # Extended the final silence to 600ms.
    # Combined with the chime, this ensures your 1-second ducker has fully
    # engaged the music volume reduction before the TTS actually starts talking.
    silence = np.zeros(int(RATE * 0.6), dtype=np.float32)
    chime = np.concatenate((chime, silence))

    padded_length = ((len(chime) + FRAMES_PER_PACKET - 1) // FRAMES_PER_PACKET) * FRAMES_PER_PACKET
    chime = np.pad(chime, (0, padded_length - len(chime)))

    return chime.reshape(-1, CHANNELS).astype(SAMPLE_FORMAT)


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
    """Drain the UDP socket directly into the shared ring buffer.

This receiver runs in a dedicated process to eliminate contention with the main
audio playback loop. On fast networks, contention can delay socket reads just
long enough for the OS-level receive buffer to overflow—even when configured to
its maximum size—resulting in dropped packets.

By isolating the socket reader, this process can drain incoming packets into the
preallocated ring buffer as quickly as they arrive. This prevents socket overruns
and data loss, while ensuring the playback thread remains responsive.
    """
    import socket, numpy as np

    def _sigterm_handler(signum, frame):
        print("Signal handler in RECEIVER called")
        raise KeyboardInterrupt("SIGTERM received from launchd in RECEIVER")
    
    signal.signal(signal.SIGTERM, _sigterm_handler)    

    setproctitle.setproctitle("Hermes Socket Receiver")
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
    """Set the AirPlay device volume to the requested level.

    This is used by the ducker to briefly reduce the AirPlay volume while the local
    Hermes audio is playing, then restore it afterward. The function uses a direct
    HTTP PUT instead of a higher-level library so it can stay lightweight and
    dependency-free.

    Args:
        level: Integer percentage from 0 to 100.
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
    """Lower AirPlay volume while local Hermes audio is active.

    The ducker is a lightweight background thread: it waits for a toggle signal from
    the playback callback, drops the AirPlay volume for the duration of local audio,
    and then restores it when the callback indicates the stream has stopped or gone
    quiet. This keeps the local stream audible without requiring a full audio mixing
    stack.
    """
    if AIRPLAY_ID is None:
        print("Not running ducker on this node")
        return # No device to duck.
    
    print("Ducker Started")
    try:
        while not exit_flag.is_set():
            if not duck_event.wait_set(timeout=0.5):
                continue
            set_airplay_volume(50)
            print("Ducking Airplay Volume")
            duck_event.wait_clear()
            print("Restoring Airplay Volume")
            set_airplay_volume(100)
    except KeyboardInterrupt:
        pass
    
    print("Ducker thread exited")
    
# Main process
# The application is launched as a standalone listener process: it owns the
# network socket, the shared memory ring, and the playback stream for the local
# speaker. Keeping the producer/consumer split explicit makes it easier to reason
# about startup, buffering, and shutdown timing.
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
    
    # Pre-allocate slice views to avoid ndarray creation during callback execution
    ring_slices = [ring[i] for i in range(RING_SIZE)]
    START_CHIME = make_start_chime()
    chime_len = len(START_CHIME)

    # Fast local bindings for closure
    mask = MASK
    start_thresh = START_THRESHOLD
    gap_seconds = UTTERANCE_GAP_SECONDS
    duck_set = duck_audio.set
    duck_clear = duck_audio.clear
    monotonic = time.monotonic

    played_packets = 0
    # The end of the array is the "not playing" sentinel. While a chime is
    # active, this is the offset of the next samples to send to the device.
    chime_position = chime_len
    quiet_since: float | None = None

    def callback(outdata, frames, callback_time, status):
        """sounddevice callback: fill each audio block from the shared ring buffer.

        The callback runs on the audio thread and is expected to be very fast. It
        checks how many packets are waiting in the ring, starts playback once the
        buffer has enough data to avoid startup stutter, plays a chime once at the
        beginning of each utterance, and emits silence when the stream has dropped out.
        """
        global playing, played_packets, chime_position, quiet_since

        # Number of packets buffered since the producer pointer outran the consumer.
        available = write_idx.value - read_idx.value

        if chime_position < chime_len:
            # Output one callback-sized slice at a time. Network audio remains
            # buffered while the chime plays, so speech starts afterward.
            # Safe to assign directly without zero-fill or min() boundary clamping because
            # make_start_chime() guarantees chime_len is an exact multiple of frames (FRAMES_PER_PACKET).
            chime_end = chime_position + frames
            outdata[:] = START_CHIME[chime_position:chime_end]
            chime_position = chime_end
            return

        if not playing:
            played_packets = 0
            if available >= start_thresh:
                playing = True
                duck_set()
                now = monotonic()
                if quiet_since is None or (now - quiet_since) >= gap_seconds:
                    # A sufficiently long quiet period marks a new utterance.
                    chime_position = frames
                    outdata[:] = START_CHIME[:frames]
                    return
            else:
                if quiet_since is None:
                    quiet_since = monotonic()
                outdata.fill(0.0)
                return

        if available > 0:
            slot = read_idx.value & mask
            outdata[:] = ring_slices[slot]
            read_idx.value += 1
            played_packets += 1
        else:
            # When the network is quiet, stop playback and let the ducker restore the
            # AirPlay volume to avoid a very audible sudden transition.
            outdata.fill(0.0)
            playing = False
            quiet_since = monotonic()
            duck_clear()
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