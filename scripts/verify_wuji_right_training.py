"""Read real train/validation samples and check the 27D LoRA parameter contract."""
import hashlib
from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.training.config import get_config
from openpi.training.data_loader import create_torch_dataset, transform_dataset

config = get_config("pi05_wuji_right_hex_lora")
data_config = config.data.create(config.assets_dirs, config.model)
root = Path(config.data.repo_id).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    digest, relative = line.split("  ", 1)
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
print("Dataset SHA256 checks: PASS", flush=True)
dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
print("TRAIN_DATASET", len(dataset), flush=True)
for index in [0, len(dataset) // 2, len(dataset) - 1]:
    sample = dataset[index]
    assert sample["observation.state"].shape == (27,)
    assert sample["action"].shape == (50, 27)
    assert sample["observation.images.cam_right_wrist"].shape == (3, 224, 224)
    print("Sample", index, "video, state, action: PASS", flush=True)
sample = transform_dataset(dataset, data_config)[0]
assert sample["actions"].shape == (50, 27)
assert not sample["image_mask"]["left_wrist_0_rgb"]
assert np.isfinite(sample["actions"]).all()
print("Normalized and tokenized 27D sample: PASS", flush=True)
validation = LeRobotDataset(str(root / "val"))
assert validation[0]["observation.state"].shape == (27,)
print("VALIDATION_DATASET", len(validation), "segments", validation.num_episodes, flush=True)
model = nnx.eval_shape(config.model.create, jax.random.key(0))
trainable = nnx.state(model).filter(config.trainable_filter).flat_state()
print("Trainable parameters", sum(int(np.prod(v.value.shape)) for v in trainable.values()), flush=True)
assert all("img" not in path for path in trainable)
assert any("action_in_proj" in path for path in trainable)
assert any("lora" in "/".join(map(str, path)) for path in trainable)
print("LoRA + action head, frozen vision: PASS", flush=True)
