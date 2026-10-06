import asyncio
import threading


class LoopThread:
    """An asyncio event loop on a daemon thread, so sync LoadGen callbacks can schedule coroutines on it."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def spawn(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro):
        return self.spawn(coro).result()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join()
