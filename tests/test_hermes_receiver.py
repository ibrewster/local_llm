import array
import struct
import sys
import unittest
from unittest.mock import MagicMock

class TestHermesReceiver(unittest.TestCase):
    def setUp(self):
        # Ensure numpy is not used by hermes_receiver
        if 'hermes_receiver' in sys.modules:
            del sys.modules['hermes_receiver']

    def test_imports_and_constants_without_numpy(self):
        # Block numpy
        original_numpy = sys.modules.get('numpy')
        sys.modules['numpy'] = None
        try:
            import hermes_receiver as hr
            self.assertEqual(hr.SAMPLE_FORMAT, 'float32')
            self.assertEqual(hr.BLOCK_BYTES, hr.CHANNELS * hr.FRAMES_PER_PACKET * 4)
            self.assertEqual(hr.RING_SIZE, 4096)
            self.assertEqual(hr.MASK, 4095)
        finally:
            if original_numpy is not None:
                sys.modules['numpy'] = original_numpy
            else:
                sys.modules.pop('numpy', None)

    def test_make_start_chime(self):
        original_numpy = sys.modules.get('numpy')
        sys.modules['numpy'] = None
        try:
            import hermes_receiver as hr
            chime = hr.make_start_chime()
            self.assertIsInstance(chime, bytes)
            self.assertEqual(len(chime) % hr.BLOCK_BYTES, 0)
            
            # Check float samples
            num_floats = len(chime) // 4
            samples = struct.unpack(f'{num_floats}f', chime)
            self.assertGreater(len(samples), 0)
            # Ensure amplitude is bounded within [-1.0, 1.0]
            max_amp = max(abs(s) for s in samples)
            self.assertLessEqual(max_amp, 1.0)
            self.assertGreater(max_amp, 0.1)
        finally:
            if original_numpy is not None:
                sys.modules['numpy'] = original_numpy
            else:
                sys.modules.pop('numpy', None)

    def test_bidirectional_event(self):
        import hermes_receiver as hr
        event = hr.BidirectionalEvent()
        self.assertFalse(event.is_set())
        
        event.set()
        self.assertTrue(event.is_set())
        self.assertTrue(event.wait_set(timeout=0.1))
        
        event.clear()
        self.assertFalse(event.is_set())
        self.assertTrue(event.wait_clear(timeout=0.1))

    def test_shared_memory_ring_views(self):
        import hermes_receiver as hr
        import multiprocessing.shared_memory as shm
        
        ring_shm = shm.SharedMemory(create=True, size=hr.RING_SIZE * hr.BLOCK_BYTES)
        try:
            buf = ring_shm.buf
            ring_slices = [buf[i * hr.BLOCK_BYTES : (i + 1) * hr.BLOCK_BYTES] for i in range(hr.RING_SIZE)]
            self.assertEqual(len(ring_slices), hr.RING_SIZE)
            
            # Write and verify slice
            test_bytes = b'\x12' * hr.BLOCK_BYTES
            ring_slices[5][:] = test_bytes
            self.assertEqual(bytes(ring_slices[5]), test_bytes)
        finally:
            for sl in ring_slices:
                sl.release()
            del ring_slices
            buf.release()
            del buf
            ring_shm.close()
            ring_shm.unlink()

if __name__ == '__main__':
    unittest.main()
