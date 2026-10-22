# Classical CV vehicle tracking
# Run `make help` to see all available targets.

SHELL := /bin/bash
export UID_GID := $(shell id -u):$(shell id -g)

COMPOSE := docker compose
RUN    := $(COMPOSE) run --rm cv
RUN_GT := $(COMPOSE) --profile gt run --rm gt

VIDEO ?= data/1900-151662242.mp4
OUT   ?= output/run
ARGS  ?=
ONNX  ?= models/yolov8s.onnx
VT_MODELS ?= ../vehicle_tracker/models

.PHONY: help setup build build-gt rebuild shell shell-gt test test-v lint format \
        track track-debug synthetic eval-synthetic models gt-detect gt gt-all eval compare clean

## —— Setup ———————————————————————————————————————

help: ## Show this help message
	@echo "Usage: make <target> [VIDEO=...] [OUT=...] [ARGS=\"...\"]"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## First-time setup: create data/output dirs, build the CPU image
	@mkdir -p data output
	@test -f $(VIDEO) || echo "Put the input video at $(VIDEO) (or pass VIDEO=...)"
	$(COMPOSE) build cv

build: ## Build the CPU image (classical tracker)
	$(COMPOSE) build cv

build-gt: ## Build the GPU image (TensorRT detector, ground truth only)
	$(COMPOSE) --profile gt build gt

rebuild: ## Force rebuild both images
	$(COMPOSE) --profile gt build --no-cache

## —— Development —————————————————————————————————

shell: ## Open a bash shell in the CPU container
	$(RUN) bash

shell-gt: ## Open a bash shell in the GPU container
	$(RUN_GT) bash

test: ## Run pytest
	$(RUN) pytest -q --tb=short

test-v: ## Run pytest verbose
	$(RUN) pytest -v --tb=short

lint: ## Run ruff check
	$(RUN) ruff check --no-cache .

format: ## Run ruff format
	$(RUN) ruff format --no-cache .

## —— Classical tracker ———————————————————————————

track: ## Track the two reference cars. Usage: make track [VIDEO=...] [OUT=output/run] [ARGS="..."]
	$(RUN) python track_vehicles.py --video $(VIDEO) --out-dir $(OUT) $(ARGS)

track-debug: ## Same as track, plus candidate boxes and scores (for tuning)
	$(RUN) python track_vehicles.py --video $(VIDEO) --out-dir $(OUT) --debug-candidates $(ARGS)

synthetic: ## Render the synthetic test clip with exact ground truth
	@mkdir -p output/synthetic
	$(RUN) python tools/make_synthetic_video.py --out output/synthetic/synthetic.mp4 --gt output/synthetic/gt.json

eval-synthetic: synthetic ## Track and score the synthetic clip
	$(RUN) python track_vehicles.py --video output/synthetic/synthetic.mp4 --out-dir output/synthetic/run --no-video
	$(RUN) python tools/evaluate.py output/synthetic/run/tracks.csv output/synthetic/gt.json

## —— Ground truth (deep detector, comparison only) ——

models: ## Copy the COCO YOLOv8s ONNX file from ../vehicle_tracker/models
	@mkdir -p models
	cp -n $(VT_MODELS)/yolov8s.onnx models/

gt-detect: ## Run the TensorRT detector on every frame -> gt/detections.csv
	$(RUN_GT) python3 tools/detect_trt.py --video $(VIDEO) --onnx $(ONNX) --out gt/detections.csv

gt: ## Link detections, pick the target tracklets -> gt/gt.json + gt/contact_sheet.jpg. ARGS="--pin NAME=ID"
	$(RUN) python tools/build_gt.py --video $(VIDEO) --detections gt/detections.csv --out-dir gt $(ARGS)

gt-all: models gt-detect gt ## models + gt-detect + gt

eval: ## Score a tracker run against gt/gt.json. Usage: make eval [OUT=output/run]
	$(RUN) python tools/evaluate.py $(OUT)/tracks.csv gt/gt.json

compare: ## Side-by-side video: classical tracker vs ground truth -> $(OUT)/compare.mp4
	$(RUN) python tools/compare_video.py --video $(VIDEO) --tracks $(OUT)/tracks.csv --gt gt/gt.json --out $(OUT)/compare.mp4

clean: ## Remove outputs and caches (keeps data/, models/ and gt/)
	rm -rf output .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
