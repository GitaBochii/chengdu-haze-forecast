"""Seekable bounded range reader for remote HDF5 metadata and small slices."""
import io
from collections import OrderedDict


class RangeFile(io.RawIOBase):
    def __init__(self, size, fetch, block_size=16384, max_bytes=8 * 1024 * 1024):
        super().__init__()
        if size <= 0 or block_size <= 0 or max_bytes <= 0:
            raise ValueError('positive sizes required')
        self.size, self.fetch, self.block_size = size, fetch, block_size
        self.max_bytes = max_bytes
        self.position = 0
        self.bytes_fetched = 0
        self.requests = 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError('invalid seek mode')
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if target < 0:
            raise ValueError('negative position')
        self.position = target
        return target

    def read(self, size=-1):
        self._checkClosed()
        end = self.size if size < 0 else min(self.size, self.position + size)
        parts = []
        while self.position < end:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                stop = min(self.size, start + self.block_size) - 1
                count = stop - start + 1
                if self.bytes_fetched + count > self.max_bytes:
                    raise ValueError('remote metadata read budget exceeded')
                body = self.fetch(start, stop)
                self.requests += 1
                self.bytes_fetched += len(body)
                if len(body) != count:
                    raise ValueError('invalid fetched block length')
                self.cache[block] = body
                if len(self.cache) > 128:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            offset = self.position % self.block_size
            length = min(end - self.position, len(self.cache[block]) - offset)
            parts.append(self.cache[block][offset:offset + length])
            self.position += length
        return b''.join(parts)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)
