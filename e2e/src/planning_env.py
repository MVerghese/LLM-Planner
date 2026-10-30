"""
Gymnasium-style environment for fruit-drawer task planning.

Action space is derived from ``object_properties.yaml``: Open/Close for each openable
object, PickupObject for each pickupable object, and PutObject for each
(pickupable, receptacle) pair.

Observation space: RGB image (uint8). With an optional PolicyExecution hook,
observations are the Franka env's ``world_image`` (480, 640, 3).

Reward: placeholder for now.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

_FRANKA_CONTROL = "/home/mverghese/franka_control"
if _FRANKA_CONTROL not in sys.path:
    sys.path.insert(0, _FRANKA_CONTROL)
from Policy_Execution import PolicyExecution, get_policy_execution  # noqa: E402

# FrankaGymEnvironment observation_space (cartesian): world_image shape
FRANKA_WORLD_IMAGE_SHAPE: Tuple[int, int, int] = (480, 640, 3)

def _e2e_config_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))


# Task label → ``object_properties.yaml`` path. Keys must match
# ``get_policy_execution`` task names where policies exist.
TASK_NAME_TO_OBJECT_PROPERTIES_YAML: Dict[str, str] = {
    "Put Away Fruit in the Drawer": os.path.join(
        _e2e_config_dir(), "fruit_object_properties.yaml"
    ),
}


def _parse_property_name_list(raw: Any, key: str, yaml_path: str) -> List[str]:
    """Parse Openable / Pickupable / Receptacle value: comma-separated string or YAML list."""
    if raw is None:
        raise KeyError(f"{key} missing in {yaml_path}")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def load_object_properties_yaml(yaml_path: str) -> Dict[str, List[str]]:
    """
    Load ``Openable``, ``Pickupable``, and ``Receptacle`` from YAML.

    ``yaml_path`` must be an explicit path (no default).
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected mapping at root of {yaml_path}, got {type(cfg)}")
    return {
        "openable": _parse_property_name_list(cfg.get("Openable"), "Openable", yaml_path),
        "pickupable": _parse_property_name_list(
            cfg.get("Pickupable"), "Pickupable", yaml_path
        ),
        "receptacle": _parse_property_name_list(
            cfg.get("Receptacle"), "Receptacle", yaml_path
        ),
    }


def build_planning_action_strings(properties: Dict[str, List[str]]) -> List[str]:
    """
    Build discrete action strings from object properties.

    - ``OpenObject`` / ``CloseObject`` for each openable object.
    - ``PickupObject`` for each pickupable object.
    - ``PutObject`` for each (pickupable object, receptacle) pair.
    """
    openable = properties["openable"]
    pickupable = properties["pickupable"]
    receptacle = properties["receptacle"]

    actions: List[str] = []
    for obj in openable:
        actions.append(f"OpenObject {obj}")
    for obj in openable:
        actions.append(f"CloseObject {obj}")
    for obj in pickupable:
        actions.append(f"PickupObject {obj}")
    for obj in pickupable:
        for rec in receptacle:
            actions.append(f"PutObject {obj} {rec}")
    return actions


def _world_image_from_obs_dict(obs: Dict[str, Any]) -> np.ndarray:
    """Extract world camera RGB (uint8 HxWx3) from Franka observation dict."""
    img = np.asarray(obs["world_image"])
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    return np.ascontiguousarray(img)


def build_planning_env(
    task_name: str,
    load_policy_execution: bool = False,
) -> PlanningEnv:
    """
    Resolve ``task_name`` to an objects YAML path, validate it loads, optionally
    attach ``PolicyExecution`` from ``get_policy_execution``, and return ``PlanningEnv``.

    Args:
        task_name: Key in ``TASK_NAME_TO_OBJECT_PROPERTIES_YAML`` (and must be supported by
            ``get_policy_execution`` when ``load_policy_execution`` is True).
        load_policy_execution: If True, ``get_policy_execution(task_name)`` is used.

    Raises:
        ValueError: Unknown ``task_name`` or ``get_policy_execution`` failure.
    """
    if task_name not in TASK_NAME_TO_OBJECT_PROPERTIES_YAML:
        raise ValueError(
            f"Unknown task_name {task_name!r}. "
            f"Known tasks: {list(TASK_NAME_TO_OBJECT_PROPERTIES_YAML.keys())}"
        )
    object_properties_yaml = TASK_NAME_TO_OBJECT_PROPERTIES_YAML[task_name]
    load_object_properties_yaml(object_properties_yaml)

    executor: Optional[PolicyExecution] = None
    if load_policy_execution:
        executor = get_policy_execution(task_name)

    return PlanningEnv(
        object_properties_yaml=object_properties_yaml,
        policy_execution=executor,
    )


class PlanningEnv(gym.Env):
    """
    Discrete planning actions; image observation from Franka ``world_image`` when
    ``policy_execution`` is set, otherwise a placeholder zero image.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        object_properties_yaml: str,
        image_shape: Tuple[int, int, int] = (256, 256, 3),
        render_mode: Optional[str] = None,
        policy_execution: Optional[PolicyExecution] = None,
        policy_num_steps: int = 200,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._object_properties = load_object_properties_yaml(object_properties_yaml)
        self._action_strings = build_planning_action_strings(self._object_properties)
        self.action_space = spaces.Discrete(len(self._action_strings))

        self._policy_execution = policy_execution
        self._policy_num_steps = int(policy_num_steps)

        if self._policy_execution is not None:
            self._image_shape = FRANKA_WORLD_IMAGE_SHAPE
        else:
            h, w, c = image_shape
            if c != 3:
                raise ValueError("image_shape must be (H, W, 3)")
            self._image_shape = image_shape

        self.observation_space = spaces.Box(
            low=0, high=255, shape=self._image_shape, dtype=np.uint8
        )
        self._last_world_image: Optional[np.ndarray] = None

    @property
    def action_strings(self) -> List[str]:
        return list(self._action_strings)

    def action_to_string(self, action: Union[int, np.integer]) -> str:
        a = int(action)
        if a < 0 or a >= len(self._action_strings):
            raise IndexError(f"action {a} out of range [0, {len(self._action_strings)})")
        return self._action_strings[a]

    def _get_obs(self) -> np.ndarray:
        """Current RGB observation (``world_image`` when using PolicyExecution)."""
        if self._last_world_image is not None:
            return self._last_world_image
        return np.zeros(self._image_shape, dtype=np.uint8)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if self._policy_execution is not None:
            obs_dict, _ = self._policy_execution.env.reset(
                seed if seed is not None else 0
            )
            self._last_world_image = _world_image_from_obs_dict(obs_dict)
        else:
            self._last_world_image = None
        return self._get_obs(), {}

    def step(
        self, action: Union[int, np.ndarray, np.integer]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = int(action.flat[0])
        else:
            action = int(action)
        action_str = self.action_to_string(action)
        reward = 0.0
        terminated = False
        truncated = False

        if self._policy_execution is not None:
            obs_dict, pe_success = self._policy_execution.run(
                action_str,
                num_steps=self._policy_num_steps,
                reset_env=True,
            )
            self._last_world_image = _world_image_from_obs_dict(obs_dict)
            obs = self._get_obs()
            info: Dict[str, Any] = {
                "action_str": action_str,
                "policy_execution_success": bool(pe_success),
            }
        else:
            obs = self._get_obs()
            info = {"action_str": action_str}

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._get_obs()
        return None

    def close(self) -> None:
        if self._policy_execution is not None:
            self._policy_execution.close()
            self._policy_execution = None
        self._last_world_image = None

def main():
    env = build_planning_env("Put Away Fruit in the Drawer")
    print(env.action_strings)

if __name__ == "__main__":
    main()