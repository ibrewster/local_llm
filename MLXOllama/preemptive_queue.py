import queue

class _Front:
    """Wrapper to signal that an item should be inserted at the head."""
    __slots__ = ('item',)
    def __init__(self, item):
        self.item = item

class PreemptiveQueue(queue.Queue):
    """queue.Queue plus put_front(), which inserts at the head (next to be got)."""

    def _put(self, item):
        if type(item) is _Front:
            self.queue.appendleft(item.item)
        else:
            super()._put(item)

    def put_front(self, item, block=True, timeout=None):
        self.put(_Front(item), block=block, timeout=timeout)

    def put_front_nowait(self, item):
        self.put_front(item, block=False)

    def snapshot(self):
        """Consistent copy of the queued items, safe to iterate."""
        with self.mutex:
            return list(self.queue)