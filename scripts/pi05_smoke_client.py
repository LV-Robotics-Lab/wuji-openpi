#!/usr/bin/env python3
"""Send one recorded validation observation to the model service; no robot commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/wuji-openpi/packages/openpi-client/src"))


def letterbox_rgb(image):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    h, w = image.shape[:2]
    if min(h, w) < 1:
        raise ValueError("Empty image")
    scale = 224 / max(h, w)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    result = np.zeros((224, 224, 3), np.uint8)
    y, x = (224 - resized.shape[0]) // 2, (224 - resized.shape[1]) // 2
    result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return result


def read_observation(dataset, episode, frame):
    import pyarrow.parquet as pq
    info = json.loads((dataset / "meta/info.json").read_text())
    fields = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
    table = pq.read_table(dataset / info["data_path"].format(**fields))
    if not 0 <= frame < len(table):
        raise ValueError(f"Frame {frame} outside episode of length {len(table)}")
    observation = {"observation/state": np.asarray(table["observation.state"][frame].as_py(), dtype=np.float32)}
    for feature, key in (("observation.images.stereo_right", "observation/image"),
                         ("observation.images.cam_right_wrist", "observation/right_wrist_image")):
        path = dataset / info["video_path"].format(**fields, video_key=feature)
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame} from {path}")
            observation[key] = letterbox_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/franka_wuji/training/hexagonal_peg_insertion/openpi_v21/val")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    observation = read_observation(args.dataset, args.episode, args.frame)
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    responses, elapsed = [], []
    with connect(args.url, compression=None, max_size=None, open_timeout=10) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
        if (metadata.get("action_space"), metadata.get("action_dim"), metadata.get("action_horizon"),
            metadata.get("action_units")) != ("franka_wuji_right_joint27", 27, 50, "absolute_joint_radians"):
            raise ValueError(f"Wrong server contract: {metadata}")
        for _ in range(3):
            start = time.monotonic()
            ws.send(msgpack_numpy.packb(observation))
            reply = ws.recv(timeout=120)
            if isinstance(reply, str):
                raise RuntimeError(reply)
            result = msgpack_numpy.unpackb(reply)
            actions = np.asarray(result["actions"])
            if actions.shape != (50, 27) or not np.isfinite(actions).all():
                raise ValueError("Invalid returned action chunk")
            elapsed.append((time.monotonic() - start) * 1000)
            responses.append(actions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, actions=np.asarray(responses), state=observation["observation/state"],
                            round_trip_ms=np.asarray(elapsed), metadata=json.dumps(metadata))
    print(json.dumps({"status": "PASS", "actions_shape": [3, 50, 27],
                      "round_trip_ms": elapsed, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
