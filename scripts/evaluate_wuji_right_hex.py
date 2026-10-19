"""Offline validation of the 27D Wuji pi05 checkpoint; never controls hardware.

Evaluate every validation anchor and a seeded training subset. Report the model's
flow-matching objective separately from sampled absolute-joint action errors.
Exclude end-of-segment action padding from action metrics, and compare against
holding the current measured joint state over the same future horizons.
"""
import argparse
import dataclasses
import json
import logging
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils, normalize
from openpi.training import config as config_lib, data_loader
from openpi import transforms


def error_metrics(predicted, target, state, valid):
    if predicted.shape != target.shape or predicted.shape[-1] != 27:
        raise ValueError("Expected matching (N,H,27) prediction and target arrays")
    if valid.shape != predicted.shape[:2] or state.shape != (len(predicted), 27):
        raise ValueError("Invalid mask or state shape")
    if not all(np.isfinite(x).all() for x in (predicted, target, state)):
        raise ValueError("Non-finite prediction, target or state")
    errors = {"policy": predicted - target, "hold_current_state": state[:, None, :] - target}
    result = {}
    for name, error in errors.items():
        horizons = {}
        for label, count in (("first_action", 1), ("first_10_actions", 10), ("full_50_actions", 50)):
            values = error[:, :count][valid[:, :count]]
            if not len(values):
                raise ValueError("No non-padding target actions")
            groups = {}
            for group, span in (("arm", slice(0, 7)), ("hand", slice(7, 27)), ("all", slice(None))):
                v = values[:, span].astype(np.float64)
                groups[group] = {"mae_rad": float(np.abs(v).mean()),
                                 "rmse_rad": float(np.sqrt(np.square(v).mean())),
                                 "p95_abs_rad": float(np.quantile(np.abs(v), .95))}
            horizons[label] = {"valid_target_steps": len(values), "groups": groups,
                               "per_joint_mae_rad": np.abs(values).mean(axis=0).tolist()}
        result[name] = horizons
    return result


def source_mapping(root, split):
    report = json.loads((root / "conversion.json").read_text())
    train_ids = {x["source_episode"] for x in report["episodes"] if x["split"] == "train"}
    val_ids = {x["source_episode"] for x in report["episodes"] if x["split"] == "val"}
    if train_ids & val_ids:
        raise ValueError("Source episode leakage between training and validation")
    return {seg["episode_index"]: ep["source_episode"]
            for ep in report["episodes"] if ep["split"] == split for seg in ep["output_episodes"]}


def evaluate_split(config, stats, loss_fn, sample_fn, split, output, batch_size, repeats, train_frames, seed):
    root = Path(config.data.repo_id).parent
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(data_config, repo_id=str(root / split), norm_stats=stats)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    pre = transforms.compose([
        *data_config.repack_transforms.inputs, *data_config.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    post = transforms.compose([
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])
    lengths = {e["episode_index"]: e["length"] for e in map(json.loads, (root / split / "meta/episodes.jsonl").read_text().splitlines())}
    source_ids = source_mapping(root, split)
    indices = np.arange(len(dataset))
    if split == "train" and train_frames < len(indices):
        indices = np.sort(np.random.default_rng(seed).choice(indices, train_frames, replace=False))
    predicted, targets, states, masks, losses, episodes, source = [], [], [], [], [], [], []
    started = time.monotonic()
    print(f"Evaluating {split}: {len(indices)} / {len(dataset)} anchors, {repeats} flow-loss draws, 10-step inference", flush=True)
    for batch_number, start in enumerate(range(0, len(indices), batch_size)):
        batch_indices = indices[start:start + batch_size]
        raw_states, raw_targets, valid_masks, processed = [], [], [], []
        for index in batch_indices:
            item = dataset[int(index)]
            episode = int(item["episode_index"])
            frame = int(item["frame_index"])
            raw_states.append(np.array(item["observation.state"], dtype=np.float32, copy=True))
            raw_targets.append(np.array(item["action"], dtype=np.float32, copy=True))
            valid_masks.append(np.arange(config.model.action_horizon) < (lengths[episode] - frame))
            episodes.append(episode)
            source.append(source_ids[episode])
            # The delta transform mutates the input action tensor; retain copies above.
            processed.append(pre(item))
        n = len(processed)
        processed += [processed[-1]] * (batch_size - n)
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *processed)
        observation = model_lib.Observation.from_dict(batch)
        repeat_losses = []
        for repeat in range(repeats):
            key = jax.random.fold_in(jax.random.key(seed + repeat), batch_number)
            repeat_losses.append(np.asarray(loss_fn(key, observation, batch["actions"], train=False))[:n])
        norm_prediction = np.asarray(sample_fn(jax.random.fold_in(jax.random.key(seed + 10000), batch_number), observation, num_steps=10))[:n]
        absolute = []
        for i in range(n):
            out = post({"state": np.array(batch["state"][i], copy=True), "actions": norm_prediction[i].copy()})
            absolute.append(out["actions"])
            # Verify output-unit reconstruction independently against the actual raw target.
            decoded_target = post({"state": np.array(batch["state"][i], copy=True),
                                   "actions": np.array(batch["actions"][i], copy=True)})["actions"]
            np.testing.assert_allclose(decoded_target, raw_targets[i], rtol=2e-5, atol=2e-5)
        predicted.extend(absolute)
        targets.extend(raw_targets)
        states.extend(raw_states)
        masks.extend(valid_masks)
        losses.extend(np.mean(repeat_losses, axis=0))
        if batch_number % 20 == 0 or start + n == len(indices):
            elapsed = time.monotonic() - started
            print(f"{split} {start+n}/{len(indices)} anchors; elapsed={elapsed:.1f}s; loss={np.mean(repeat_losses):.6f}", flush=True)
    predicted, targets, states, masks, losses = map(np.asarray, (predicted, targets, states, masks, losses))
    if not np.isfinite(losses).all():
        raise ValueError("Non-finite flow loss")
    metrics = error_metrics(predicted, targets, states, masks)
    source = np.asarray(source)
    per_episode = {}
    for source_id in sorted(set(source)):
        select = source == source_id
        per_episode[str(source_id)] = {
            "anchors": int(select.sum()), "flow_matching_loss": float(losses[select].mean()),
            "action_metrics": error_metrics(predicted[select], targets[select], states[select], masks[select]),
        }
    result = {"split": split, "evaluated_anchors": len(indices), "total_dataset_anchors": len(dataset),
              "source_episodes": len(per_episode), "flow_matching_loss": float(losses.mean()),
              "flow_matching_loss_excluding_padding": float(losses[masks].mean()),
              "action_metrics": metrics, "per_source_episode": per_episode,
              "elapsed_s": time.monotonic() - started}
    np.savez_compressed(output / f"{split}_predictions.npz", indices=indices, predicted=predicted,
                        targets=targets, states=states, valid_mask=masks, flow_loss=losses,
                        episode_index=np.asarray(episodes), source_episode=source)
    (output / f"{split}_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"FINISHED {split}: flow_loss={result['flow_matching_loss']:.6f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loss-repeats", type=int, default=2)
    parser.add_argument("--train-frames", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if min(args.batch_size, args.loss_repeats, args.train_frames) < 1:
        raise ValueError("Batch size, repetitions and training sample size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    config = config_lib.get_config("pi05_wuji_right_hex_lora")
    metadata = json.loads((args.checkpoint / "_CHECKPOINT_METADATA").read_text())
    assert metadata["commit_timestamp_nsecs"] is not None
    print("Restoring final checkpoint at saved parameter precision", flush=True)
    model = config.model.load(model_lib.restore_params(args.checkpoint / "params"))
    model.eval()
    stats = normalize.load(args.checkpoint / "assets" / config.data.assets.asset_id)
    loss_fn = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))
    sample_fn = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    results = {}
    for split in ("val", "train"):
        results[split] = evaluate_split(config, stats, loss_fn, sample_fn, split, args.output,
                                       args.batch_size, args.loss_repeats, args.train_frames, args.seed)
    result = {"checkpoint": str(args.checkpoint), "seed": args.seed, "batch_size": args.batch_size,
              "loss_repeats": args.loss_repeats, "inference_flow_steps": 10,
              "parameter_precision": "original checkpoint (BF16 frozen weights, FP32 trainable weights)",
              "image_augmentation": False, "action_metrics_exclude_padding": True,
              "loss_ratio_val_to_train": results["val"]["flow_matching_loss"] / results["train"]["flow_matching_loss"],
              "results": results}
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
