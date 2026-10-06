"""CLI entry: run one model config (configs/<model>/<scenario>_<system>.yaml) through LoadGen."""
import argparse
import pathlib

import mlperf_loadgen as lg

from vsora_mlperf.backends import create_backend
from vsora_mlperf.datasets import create_dataset
from vsora_mlperf.harness.config import load_config
from vsora_mlperf.harness.offline_sut import OfflineSUT
from vsora_mlperf.harness.server_sut import ServerSUT

LG_SCENARIOS = {"Offline": lg.TestScenario.Offline, "Server": lg.TestScenario.Server}
LG_MODES = {"performance": lg.TestMode.PerformanceOnly, "accuracy": lg.TestMode.AccuracyOnly}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="e.g. configs/smollm2-135m-instruct/offline_cpu.yaml")
    p.add_argument(
        "overrides",
        nargs="*",
        help="dotted key=value, e.g. scenario=server test_mode=accuracy backend_config.device=cuda",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    hc = cfg["harness_config"]
    out_dir = pathlib.Path(hc["output_log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = create_backend(cfg["backend"], **cfg["backend_config"])
    backend.load(cfg["scenario"])
    dataset = create_dataset(cfg["dataset_config"], backend.encode)
    qsl = lg.ConstructQSL(
        dataset.total_sample_count,
        dataset.perf_count,
        dataset.LoadSamplesToRam,
        dataset.UnloadSamplesFromRam,
    )
    settings = lg.TestSettings()
    settings.scenario = LG_SCENARIOS[cfg["scenario"]]
    settings.mode = LG_MODES[cfg["test_mode"]]
    settings.FromConfig(hc["user_conf_path"], cfg["benchmark_name"], cfg["scenario"])

    max_new_tokens = cfg["sampling_config"]["max_new_tokens"]
    if cfg["scenario"] == "Offline":
        sut = OfflineSUT(backend, dataset.input_ids, max_new_tokens)
    else:
        sut = ServerSUT(backend, dataset.input_ids, max_new_tokens, report_first_token=settings.use_token_latencies)

    log_out = lg.LogOutputSettings()
    log_out.outdir = str(out_dir)
    log_out.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_out

    try:
        lg.StartTestWithLogSettings(sut.sut, qsl, settings, log_settings)
    finally:
        sut.close()
        lg.DestroyQSL(qsl)
        backend.close()
    print(f"logs: {out_dir}")


if __name__ == "__main__":
    main()
