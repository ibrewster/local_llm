import logging
import socket
import time
import threading

from queue import Queue

import mlx
import numpy as np

class AudioPacketizer:
    def __init__(self, blocksize=1024, port=5005):
        self._sent_packets = 0
        self.blocksize = blocksize
        self._buffer = np.empty(0, dtype=np.float32)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
        self._port = port
        self._next_send = time.monotonic_ns()
        packet_rate = (24000 / blocksize) * 6
        self._packet_interval = int(1e9 / packet_rate)

    def push(self, audio: np.ndarray) -> None:
        """Feed audio samples, returns list of full blocksize packets."""
        self._buffer = np.concatenate([self._buffer, audio.flatten()])
        while len(self._buffer) >= self.blocksize:
            self._sendPacket(self._buffer[:self.blocksize])
            self._buffer = self._buffer[self.blocksize:]

    def flush(self) -> None:
        """Pad and emit the final partial block. Call after last TTS chunk."""
        if len(self._buffer) == 0:
            return None
        padded = np.pad(self._buffer, (0, self.blocksize - len(self._buffer)))
        self._sendPacket(padded)
        logging.info(f"Sent {self._sent_packets} packets")
        self._sent_packets = 0
        self._buffer = np.empty(0, dtype=np.float32)
    
    def _sendPacket(self, packet: np.ndarray) -> None:
        now = time.monotonic_ns()
        if now < self._next_send:
            time.sleep((self._next_send - now) / 1e9)
        else:
            self._next_send = now
            
        self._sock.sendto(packet.data, ("239.0.0.1", self._port))
        self._sent_packets += 1
        
        self._next_send += self._packet_interval
        
class AudioStreamer:
    def __init__(self, blocksize=1024, port=5005):
        logging.info(f"Starting audio streamer")
        self._queue = Queue()
        self._packetizer = AudioPacketizer(blocksize=blocksize, port=port)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        logging.info("Audio streaming thread started")
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == "FLUSH":
                self._packetizer.flush()
            elif isinstance(msg, (np.ndarray, mlx.core.array)):
                self._packetizer.push(msg)

    def push(self, audio: np.ndarray) -> None:
        self._queue.put(audio)

    def flush(self) -> None:
        self._queue.put("FLUSH")
    