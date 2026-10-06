# Entry points mirror closed/NVIDIA and closed/AMD: one Makefile, one self-contained <model>/ dir per model.
#   make run MODEL=smollm2-135m-instruct SCENARIO=server SYSTEM=cpu ARGS="dataset_config.args.total_sample_count=16"
PY := .venv/bin/python
MODEL ?= smollm2-135m-instruct
SYSTEM ?= cpu
SCENARIO ?= offline
CFG ?= $(MODEL)/$(SCENARIO)_$(SYSTEM).yaml
SMOKE_CFG := $(MODEL)/smoke_$(SYSTEM).yaml
ARGS ?=
RUN := $(PY) -m vsora_mlperf.harness.runner --config

.PHONY: setup data models run accuracy smoke-offline smoke-server smoke-all clean

setup:
	bash scripts/setup_env.sh

data:
	bash scripts/download_cnndm.sh

models:
	@for d in */; do ls $$d*.yaml >/dev/null 2>&1 && echo "$${d%/}: $$(cd $$d && ls | tr '\n' ' ')"; done; true

run:
	$(RUN) $(CFG) $(ARGS)

# Accuracy run, then the benchmark's official scorer from the config's `accuracy_config:` section.
accuracy:
	$(RUN) $(CFG) test_mode=accuracy $(ARGS)
	$(PY) -m vsora_mlperf.harness.accuracy --config $(CFG) $(ARGS)

smoke-offline:
	$(RUN) $(SMOKE_CFG) scenario=offline $(ARGS)

smoke-server:
	$(RUN) $(SMOKE_CFG) scenario=server $(ARGS)

smoke-all: smoke-offline smoke-server

clean:
	rm -rf results
