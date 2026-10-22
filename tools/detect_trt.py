#!/usr/bin/env python3
"""Run a YOLOv8 vehicle detector (TensorRT) on every frame of a video.

This is only used to build pseudo ground truth for evaluating the classical
tracker. The tracker itself never sees these detections.

Writes one row per detection: frame, x, y, w, h, conf (original pixels).
The engine is built from the ONNX file on first run and cached next to it.
"""

from __future__ import annotations

import argparse
import csv
import os

import cupy as cp
import cv2
import numpy as np
import tensorrt as trt

SIZE = 640


def build_engine(onnx_path: str, engine_path: str) -> None:
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, log)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise SystemExit("TensorRT engine build failed")
    with open(engine_path, "wb") as f:
        f.write(data)


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    s = SIZE / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), s, top, left


def decode(out: np.ndarray, conf: float, iou: float, vehicle_classes: list[int] | None):
    preds = out[0].T  # (anchors, 4 + classes)
    scores_all = preds[:, 4:]
    if vehicle_classes is not None:
        scores_all = scores_all[:, vehicle_classes]
    scores = scores_all.max(axis=1)
    keep = scores > conf
    boxes, scores = preds[keep, :4], scores[keep]
    xywh = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                            boxes[:, 2], boxes[:, 3]])
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    idx = np.array(idx, dtype=int).reshape(-1)
    return xywh[idx], scores[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--onnx", default="models/yolov8s.onnx")
    ap.add_argument("--out", default="gt/detections.csv")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    engine_path = os.path.splitext(a.onnx)[0] + f".trt{trt.__version__}.engine"
    if not os.path.exists(engine_path):
        print(f"building {engine_path} (TensorRT {trt.__version__}) ...", flush=True)
        build_engine(a.onnx, engine_path)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    out_shape = tuple(engine.get_tensor_shape(out_name))
    # Single-class fine-tuned model: (1, 5, N). COCO model: (1, 84, N), keep car/motorcycle/bus/truck.
    vehicle_classes = None if out_shape[1] == 5 else [2, 3, 5, 7]
    d_in = cp.empty((1, 3, SIZE, SIZE), cp.float32)
    d_out = cp.empty(out_shape, cp.float32)
    ctx.set_tensor_address(in_name, d_in.data.ptr)
    ctx.set_tensor_address(out_name, d_out.data.ptr)
    stream = cp.cuda.Stream()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {a.video!r}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "x", "y", "w", "h", "conf"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x, s, top, left = letterbox(frame)
            d_in.set(x, stream=stream)
            ctx.execute_async_v3(stream.ptr)
            out = d_out.get(stream=stream)
            stream.synchronize()
            boxes, scores = decode(out, a.conf, a.iou, vehicle_classes)
            for (bx, by, bw, bh), sc in zip(boxes, scores):
                bx, by, bw, bh = (bx - left) / s, (by - top) / s, bw / s, bh / s
                wr.writerow([n, round(bx), round(by), round(bw), round(bh), f"{sc:.3f}"])
            n += 1
            if n % 500 == 0:
                print(f"frame {n}", flush=True)
    print(f"{n} frames -> {a.out}")


if __name__ == "__main__":
    main()
