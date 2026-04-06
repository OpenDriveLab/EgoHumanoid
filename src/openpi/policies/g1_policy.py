"""Policy transforms for the Pika robot."""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Inputs for the G1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [0]
    - actions: [action_horizon, 12]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("head_left",)

    rename_map = {
        "head_left": "base_0_rgb",
    }
    
    # if set all state to zeros
    mask_state: bool = False

    mask_zero_action: bool = True

    only_mask_zero: bool = False



    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0

        # Pad the proprioceptive input to the action dimension of the model
        if "state" in data:
            state = transforms.pad_to_dim(data["state"], self.action_dim)
            # Ensure state has correct shape [batch_size, state_dim]
            state = state.squeeze()
        else:
            state = np.zeros(self.action_dim)

        # Parse images to uint8 (H,W,C) since LeRobot automatically stores as float32 (C,H,W)
        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                img = data["images"][camera]
                # Convert torch tensor to numpy array if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Ensure image is in uint8 format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                # Convert from [C,H,W] to [H,W,C] if needed
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[self.rename_map[camera]] = img
            else:
                raise ValueError(f"Camera {camera} not found in data")

        images["left_wrist_0_rgb"] = np.zeros_like(images["base_0_rgb"])
        images["right_wrist_0_rgb"] = np.zeros_like(images["base_0_rgb"])
        # Create image mask based on available cameras
        image_mask = {self.rename_map[camera]: np.True_ for camera in self.EXPECTED_CAMERAS}
        image_mask["left_wrist_0_rgb"] = np.False_
        image_mask["right_wrist_0_rgb"] = np.False_
        # filter unnormal state / action value, set to 0
        # state = np.where(state > np.pi, 0, state)
        # state = np.where(state < -np.pi, 0, state)

        # Prepare inputs dictionary
        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": masked_state,
        }

        # Add actions if present
        if "actions" in data:
            original_actions = np.asarray(data["actions"])
            original_action_dim = original_actions.shape[-1]
            
            # Pad actions to model's action_dim
            actions = transforms.pad_to_dim(original_actions, self.action_dim)
            inputs["actions"] = actions.squeeze()
            
            # Handle action_mask: extend existing mask or create new one
            if "action_mask" in data:
                # Extend existing mask with False for padding dimensions
                original_mask = np.asarray(data["action_mask"])
                if original_action_dim < self.action_dim:
                    # Pad mask with False for padding dimensions
                    pad_shape = list(original_mask.shape)
                    pad_shape[-1] = self.action_dim - original_action_dim
                    padding_mask = np.zeros(pad_shape, dtype=bool)
                    action_mask = np.concatenate([original_mask, padding_mask], axis=-1)
                else:
                    action_mask = original_mask
                inputs["action_mask"] = action_mask
            elif mask_padding:
                # Create action mask if not provided (backward compatibility)
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[..., original_action_dim:] = False
                inputs["action_mask"] = action_mask


            if not self.mask_zero_action and "action_mask" in data:
                action_mask[18:] = True

            if self.only_mask_zero and "action_mask" in data:
                action_mask[18:] = True
                action_mask[18:] = False

            

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Outputs for the Pika policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :18])} 





