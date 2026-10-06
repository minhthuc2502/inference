import mlperf_loadgen as lg
import numpy as np


def complete(query_id, tokens):
    buf = np.asarray(tokens, dtype=np.int32)
    lg.QuerySamplesComplete([lg.QuerySampleResponse(query_id, buf.ctypes.data, buf.nbytes, len(tokens))])


def first_token(query_id, token):
    """TTFT: must equal the first token of the sample's final output (TEST06)."""
    buf = np.asarray([token], dtype=np.int32)
    lg.FirstTokenComplete([lg.QuerySampleResponse(query_id, buf.ctypes.data, buf.nbytes, 1)])
