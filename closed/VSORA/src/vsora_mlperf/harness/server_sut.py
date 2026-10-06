import mlperf_loadgen as lg

from vsora_mlperf.harness.responses import complete, first_token


class ServerSUT:
    """Server = latency. Samples arrive one by one (Poisson); each goes to the backend as its own request, so its
    first token (TTFT) and completion (TPOT) are reported as soon as they exist, independently of other samples."""

    def __init__(self, backend, input_ids, max_new_tokens, report_first_token):
        self._backend = backend
        self._input_ids = input_ids
        self._max_new_tokens = max_new_tokens
        self._report_first_token = report_first_token
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        for s in query_samples:
            self._backend.submit(
                self._input_ids[s.index],
                self._max_new_tokens,
                on_complete=lambda tokens, qid=s.id: complete(qid, tokens),
                on_first_token=(lambda token, qid=s.id: first_token(qid, token)) if self._report_first_token else None,
            )

    def flush_queries(self):
        pass

    def close(self):
        lg.DestroySUT(self.sut)
