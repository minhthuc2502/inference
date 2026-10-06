"""Score an accuracy run with the benchmark's official MLPerf script (the yaml `accuracy_config:` section)."""
import argparse
import pathlib
import subprocess
import sys

from vsora_mlperf.harness.config import load_config

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="same dotted key=value overrides as the runner")
    args = p.parse_args(argv)

    cfg = load_config(args.config, [*args.overrides, "test_mode=accuracy"])
    hc = cfg["harness_config"]
    acc = cfg["accuracy_config"]
    fields = {
        "log": (pathlib.Path(hc["output_log_dir"]) / "mlperf_log_accuracy.json").resolve(),
        "dataset": pathlib.Path(cfg["dataset_config"]["args"]["dataset_path"]).resolve(),
        "model": cfg["backend_config"]["model"],
    }
    script = pathlib.Path(acc["script"])
    cmd = [sys.executable, script.name, *(str(a).format(**fields) for a in acc["args"])]
    # Official scripts import sibling modules (e.g. dataset.py), so run from their directory.
    sys.exit(subprocess.call(cmd, cwd=script.parent))


if __name__ == "__main__":
    main()
