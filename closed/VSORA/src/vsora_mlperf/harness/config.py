"""Per-model run config: <model>/<scenario>_<system>.yaml plus dotted `key.sub=value` CLI overrides.

Paths inside `harness_config.user_conf_path`, `dataset_config.script` and `accuracy_config.script` are relative to the yaml file, so a
model directory is self-contained.
"""
import pathlib

import yaml

SCENARIOS = ("Offline", "Server")
TEST_MODES = ("performance", "accuracy")


def load_config(path, overrides=()):
    path = pathlib.Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must be key=value")
        *parents, leaf = key.split(".")
        node = cfg
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)

    cfg["scenario"] = cfg["scenario"].capitalize()
    cfg.setdefault("test_mode", "performance")
    if cfg["scenario"] not in SCENARIOS or cfg["test_mode"] not in TEST_MODES:
        raise ValueError(f"scenario must be one of {SCENARIOS}, test_mode one of {TEST_MODES}")
    hc = cfg["harness_config"]
    hc["user_conf_path"] = str(path.parent / hc["user_conf_path"])
    for section in ("dataset_config", "accuracy_config"):
        if "script" in cfg.get(section, {}):
            cfg[section]["script"] = str(path.parent / cfg[section]["script"])
    hc.setdefault(
        "output_log_dir",
        f"results/{path.parent.name}/{cfg['benchmark_name']}/{cfg['scenario']}/{cfg['test_mode']}",
    )
    return cfg
