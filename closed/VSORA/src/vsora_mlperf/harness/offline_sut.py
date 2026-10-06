import os
import threading
import traceback

import mlperf_loadgen as lg

from vsora_mlperf.harness.responses import complete


class OfflineSUT:
    """Offline = throughput. LoadGen issues the whole sample set in one call; it is sorted by prompt length (less
    padding for static-batch backends) and handed to the backend at once so it can batch as it likes. No
    first-token reporting: Offline scores tokens/s only."""

    def __init__(self, backend, input_ids, max_new_tokens):
        self._backend = backend
        self._input_ids = input_ids
        self._max_new_tokens = max_new_tokens
        self._threads = []
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        # Run off the LoadGen thread so issue_queries returns immediately.
        samples = sorted(query_samples, key=lambda s: len(self._input_ids[s.index]))
        self._threads.append(threading.Thread(target=self._run, args=(samples,), daemon=True))
        self._threads[-1].start()

    def _run(self, samples):
        try:
            outputs = self._backend.generate([self._input_ids[s.index] for s in samples], self._max_new_tokens)
            for s, tokens in zip(samples, outputs):
                complete(s.id, tokens)
        except BaseException:
            # LoadGen would otherwise wait forever for these samples.
            traceback.print_exc()
            os._exit(1)

    def flush_queries(self):
        pass

    def close(self):
        for t in self._threads:
            t.join()
        lg.DestroySUT(self.sut)
