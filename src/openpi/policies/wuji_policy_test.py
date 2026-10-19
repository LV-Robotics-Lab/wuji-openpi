import dataclasses

import numpy as np
import pytest

from openpi import transforms
from openpi.models.model import ModelType
from openpi.policies.wuji_policy import WujiInputs
from openpi.training.config import get_config


def test_right_training_mapping_delta_round_trip_and_mask():
    config = get_config("pi05_wuji_right_hex_lora")
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = {
        "observation.state": np.arange(27, dtype=np.float32),
        "observation.images.stereo_right": np.zeros((224, 224, 3), np.uint8),
        "observation.images.cam_right_wrist": np.ones((224, 224, 3), np.uint8),
        "action": np.ones((50, 27), np.float32),
        "prompt": "Insert the hexagonal peg.",
    }
    x = transforms.compose((*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs))(raw)
    assert x["actions"].shape == (50, 27)
    assert not x["image_mask"]["left_wrist_0_rgb"]
    assert x["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_array_equal(x["actions"][0, :7], 1 - np.arange(7))
    assert np.all(x["actions"][:, 7:] == 1)
    restored = transforms.compose(data_config.data_transforms.outputs)(x)
    np.testing.assert_allclose(restored["actions"], raw["action"])
    with pytest.raises(ValueError, match="action_dim"):
        config.data.create(config.assets_dirs, dataclasses.replace(config.model, action_dim=54))


def test_dual_mode_requires_both_cameras_and_preserves_54d():
    transform = WujiInputs(ModelType.PI05)
    data = {"observation/state": np.zeros(54), "observation/image": np.zeros((224, 224, 3), np.uint8),
            "observation/right_wrist_image": np.zeros((224, 224, 3), np.uint8)}
    with pytest.raises(KeyError):
        transform(data)
    data["observation/left_wrist_image"] = np.zeros((224, 224, 3), np.uint8)
    result = transform(data)
    assert result["state"].shape == (54,)
    assert all(result["image_mask"].values())
