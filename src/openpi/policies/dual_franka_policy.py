import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_dual_franka_example() -> dict:
    """Creates a random input example for the DualFranka policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DualFrankaInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Pad the proprioceptive input to the action dimension of the model
        # remove the velocity of rpy
        # state = np.concatenate([data["state"][..., :16], data["state"][..., 19:35]], axis=-1)
        state = data["state"]
        state = transforms.pad_to_dim(state, self.action_dim)

        mask_padding = self.model_type == _model.ModelType.PI0

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (_parse_image(data['images'][name]) for name in names)
                image_masks = (np.True_, np.True_, np.True_)
            # case _model.ModelType.PI0_FAST:
            #     names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            #     # We don't mask out padding images for FAST models.
            #     images = (base_image, np.zeros_like(base_image), wrist_image)
            #     image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state.squeeze(),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Add actions if present
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            if mask_padding:
                # Create action mask for padding
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[:, 14:] = False
                inputs["action_mask"] = action_mask
            inputs["actions"] = actions.squeeze()


        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DualFrankaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :14])}
