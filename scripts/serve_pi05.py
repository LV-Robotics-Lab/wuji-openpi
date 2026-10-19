#!/usr/bin/env python3
"""Serve the trained right-arm/hand policy. Owns no robot or sensor resources."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "data/franka_wuji/training/hexagonal_peg_insertion/checkpoints/pi05_wuji_right_hex_lora/right_hex_lora_20260906/29999"
PROMPT = "Insert the hexagonal peg into the matching hole using the right hand."
METADATA = {
    "config": "pi05_wuji_right_hex_lora", "action_space": "franka_wuji_right_joint27",
    "action_dim": 27, "action_horizon": 50, "dataset_fps": 30,
    "action_units": "absolute_joint_radians", "image_format": "RGB uint8 HWC 224x224 letterbox",
    "joint_order": [f"arm_B.joint_{i}.position_rad" for i in range(1, 8)] + [
        f"right_hand.{finger}_joint_{i}.position_rad"
        for finger in ("thumb", "index", "middle", "ring", "pinky") for i in range(4)
    ],
}


class CheckedPolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        state = np.asarray(observation["observation/state"])
        if state.shape != (27,) or not np.isfinite(state).all():
            raise ValueError("Expected 27 finite joint positions in radians")
        for key in ("observation/image", "observation/right_wrist_image"):
            image = np.asarray(observation[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{key}: expected letterboxed RGB uint8 (224,224,3)")
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"])
        if actions.shape != (50, 27) or not np.isfinite(actions).all():
            raise ValueError("Policy returned invalid actions; expected finite (50,27)")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true", help="Check assets and JAX GPU without loading weights")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = json.loads((checkpoint / "_CHECKPOINT_METADATA").read_text())
    if metadata.get("commit_timestamp_nsecs") is None:
        raise ValueError("Checkpoint has no completed commit")
    for path in (checkpoint / "params/_METADATA",
                 checkpoint / "assets/wuji_right_hex_27d/norm_stats.json",
                 Path(os.environ["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"):
        if not path.is_file():
            raise FileNotFoundError(path)
    import jax
    devices = jax.devices("gpu")  # Explicit GPU requirement; do not silently use CPU.
    print(f"GPU devices: {devices}", flush=True)
    if args.check:
        print(f"CHECK_OK: {checkpoint}", flush=True)
        return
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies.policy import Policy
    from openpi.shared import normalize
    from openpi.training import config as config_lib
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    config = config_lib.get_config("pi05_wuji_right_hex_lora")
    model = config.model.load(model_lib.restore_params(checkpoint / "params"))
    model.eval()
    data = config.data.create(config.assets_dirs, config.model)
    stats = normalize.load(checkpoint / "assets" / data.asset_id)
    policy = CheckedPolicy(Policy(
        model,
        transforms=[transforms.InjectDefaultPrompt(PROMPT), *data.data_transforms.inputs,
                    transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
                    *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,
                           transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        sample_kwargs={"num_steps": 10},
    ))
    print("Warming up inference before opening the server...", flush=True)
    policy.infer({"observation/state": np.asarray(stats["state"].mean, dtype=np.float32),
                  "observation/image": np.zeros((224, 224, 3), np.uint8),
                  "observation/right_wrist_image": np.zeros((224, 224, 3), np.uint8)})
    print(f"MODEL_READY: ws://{args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
                          metadata={**METADATA, "checkpoint_step": checkpoint.name}).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
