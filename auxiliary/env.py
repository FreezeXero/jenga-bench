"""JengaBench environment with deterministic static-tower observations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bench_common.env_sdk.base import BaseEnv, StepResult

if TYPE_CHECKING:
    from jenga.sim import JengaSimulation

PNG_CONTENT_TYPE = "image/png"
PERFECT_RAW_SCORE = 98.0
VIEWPOINT_LIMIT = 10
INVALID_ACTION_PENALTY = -0.5
VIEWPOINT_TIMEOUT_PENALTY = -10.0

DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
DISTANCES = {"Close": 15.0, "Medium": 30.0, "Full": 45.0}

@dataclass
class CameraState:
    direction: str = "SW"
    elevation_layer: int = 9
    distance: str = "Full"
    target_layer: int | None = None
    target_color: str | None = None


class JengaBenchEnv(BaseEnv):
    """Image-producing environment backed by one authoritative PyBullet client."""

    def __init__(self) -> None:
        self._camera = CameraState()
        self._consecutive_viewpoints = 0
        self._raw_points = 0.0
        self._successful_extractions = 0
        self._terminated = False
        self._termination_reason = ""
        self._seed: int | None = None
        self._simulation: JengaSimulation | None = None
        self._action_log: list[dict[str, str]] = []

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        del params
        from jenga.sim import JengaSimulation

        self.close()
        self._camera = CameraState()
        self._consecutive_viewpoints = 0
        self._raw_points = 0.0
        self._successful_extractions = 0
        self._terminated = False
        self._termination_reason = ""
        self._action_log = []
        self._seed = seed
        self._simulation = JengaSimulation()
        try:
            self._simulation.reset(seed)
        except Exception:
            self.close()
            raise
        return {
            "data": self._render_png(),
            "content_type": PNG_CONTENT_TYPE,
            "system_prompt": self._system_prompt(),
        }

    def step(self, action: Any) -> StepResult:
        if self._terminated:
            self._log_action(action, "invalid_action: episode already terminated")
            return self._result(
                reward=INVALID_ACTION_PENALTY,
                event="invalid_action: episode already terminated",
            )

        if isinstance(action, dict) and action.get("type") == "Push":
            return self._step_push(action)
        if isinstance(action, dict) and action.get("type") == "PlaceBack":
            return self._step_place_back(action)

        error = self._validate_change_viewpoint(action)
        if error is not None:
            self._raw_points += INVALID_ACTION_PENALTY
            self._log_action(action, f"invalid_action: {error}")
            return self._result(
                reward=INVALID_ACTION_PENALTY,
                event=f"invalid_action: {error}",
            )

        target_block = action.get("target_block")
        self._camera = CameraState(
            direction=action["direction"],
            elevation_layer=int(action["elevation_layer"]),
            distance=action["distance"],
            target_layer=int(target_block["layer"]) if target_block else None,
            target_color=target_block.get("color") if target_block else None,
        )
        self._consecutive_viewpoints += 1

        reward = 0.0
        event = "viewpoint_changed"
        if self._consecutive_viewpoints >= VIEWPOINT_LIMIT:
            reward = VIEWPOINT_TIMEOUT_PENALTY
            self._raw_points += reward
            self._terminated = True
            self._termination_reason = "viewpoint_timeout"
            event = "viewpoint_timeout"

        self._log_action(action, event)
        return self._result(reward=reward, event=event)

    def _step_push(self, action: dict[str, Any]) -> StepResult:
        if self._simulation is None:
            raise RuntimeError("Call reset() before stepping")
        from jenga.sim import PushRequest, PushValidationError

        try:
            request = PushRequest(
                layer=action.get("layer"),
                color=action.get("color"),
                face=action.get("face"),
                contact=action.get("contact"),
                intensity=action.get("intensity"),
            )
            push_result = self._simulation.push(request)
        except PushValidationError as exc:
            self._raw_points += INVALID_ACTION_PENALTY
            self._log_action(action, f"invalid_action: {exc}")
            return self._result(reward=INVALID_ACTION_PENALTY, event=f"invalid_action: {exc}")

        self._consecutive_viewpoints = 0
        reward = 0.0
        if push_result.outcome == "extracted":
            reward = 1.0
            self._raw_points += reward
            self._successful_extractions += 1
        self._terminated = push_result.outcome == "collapse" or self._successful_extractions >= PERFECT_RAW_SCORE
        self._termination_reason = push_result.outcome if self._terminated else ""
        if self._successful_extractions >= PERFECT_RAW_SCORE:
            self._termination_reason = "perfect_completion"
        event = f"push_{push_result.outcome}"
        self._log_action(action, event)
        return self._result(reward=reward, event=event)

    def _step_place_back(self, action: dict[str, Any]) -> StepResult:
        if self._simulation is None:
            raise RuntimeError("Call reset() before stepping")
        from jenga.sim import PlaceRequest, PlaceValidationError

        try:
            request = PlaceRequest(
                position=action.get("position"),
            )
            place_result = self._simulation.place_back(request)
        except PlaceValidationError as exc:
            self._raw_points += INVALID_ACTION_PENALTY
            self._log_action(action, f"invalid_action: {exc}")
            return self._result(reward=INVALID_ACTION_PENALTY, event=f"invalid_action: {exc}")

        self._consecutive_viewpoints = 0
        self._terminated = place_result.outcome == "collapse"
        self._termination_reason = place_result.outcome if self._terminated else ""
        event = f"place_back_{place_result.outcome}"
        self._log_action(action, event)
        return self._result(reward=0.0, event=event)

    def render(self, mode: str = "rgb_array") -> Any:
        if mode in ("rgb_array", "png"):
            return self._render_png()
        return self._camera_info()

    def _log_action(self, action: Any, event: str) -> None:
        saw = (
            f"direction={self._camera.direction}, "
            f"elevation_layer={self._camera.elevation_layer}, "
            f"distance={self._camera.distance}"
        )
        did = json.dumps(action) if isinstance(action, dict) else str(action)
        self._action_log.insert(0, {"saw": saw, "did": did, "why": event})
        self._action_log = self._action_log[:5]

    def close(self) -> None:
        if self._simulation is not None:
            self._simulation.close()
            self._simulation = None

    def _result(self, *, reward: float, event: str) -> StepResult:
        info = {
            "raw_points": self._format_score(self._raw_points),
            "normalized_score": self._format_score(self._normalized_score()),
            "blocks_removed": str(self._successful_extractions),
            "phase": self._simulation.phase if self._simulation is not None else "",
            "available_placement_positions": json.dumps(self._available_placement_positions()),
            "camera_state": json.dumps(self._camera_info(), sort_keys=True),
            "events": json.dumps([event]),
            "termination_reason": self._termination_reason,
            "action_log": json.dumps(self._action_log),
        }
        if self._simulation is not None:
            info["outcome"] = self._last_outcome()
            info["frame_count"] = str(len(self._simulation.last_frames))
            info["tower_state"] = json.dumps(self._tower_state(), sort_keys=True)
            info["replay_frames"] = json.dumps(self._simulation.last_frames, sort_keys=True)
        result = StepResult(
            observation=self._render_png(),
            reward=reward,
            terminated=self._terminated,
            truncated=False,
            info=info,
            system_prompt=self._system_prompt(),
        )
        # StepResult.content_type is available in the pinned Mesocosm revision.
        # setattr also keeps this module importable with the older public wheel.
        result.content_type = PNG_CONTENT_TYPE
        return result

    def _system_prompt(self) -> str:
        return (
            "You are playing JengaBench. Use the image as the current camera view. "
            "Return exactly one JSON action. Available actions: "
            '{"type":"ChangeViewpoint","direction":"N|NE|E|SE|S|SW|W|NW",'
            '"elevation_layer":1..18,"distance":"Close|Medium|Full",'
            '"target_block":{"layer":int,"color":"Blue|Brown|Red"}} or '
            '{"type":"Push","layer":"1..one below current top layer","color":"Blue|Brown|Red",'
            '"face":"North|South|East|West","contact":"center|left|right",'
            '"intensity":"Gentle|Firm|Hard"} or '
            '{"type":"PlaceBack","position":"Left|Middle|Right"}. '
            f"Camera: direction={self._camera.direction}, "
            f"elevation_layer={self._camera.elevation_layer}, "
            f"distance={self._camera.distance}. "
            f"Consecutive viewpoints: {self._consecutive_viewpoints}/{VIEWPOINT_LIMIT}. "
            f"Phase: {self._simulation.phase if self._simulation is not None else 'push'}. "
            f"Successful extractions: {self._successful_extractions}. "
            f"Push layer range: 1..{self._simulation.max_push_layer if self._simulation is not None else 17}. "
            f"Available placement positions: {', '.join(self._available_placement_positions()) or 'none'}. "
            "The tenth consecutive viewpoint terminates the episode with a -10 point penalty. "
            f"Action log (most recent first): {json.dumps(self._action_log) if self._action_log else 'empty'}."
        )

    def _camera_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "direction": self._camera.direction,
            "elevation_layer": self._camera.elevation_layer,
            "distance": self._camera.distance,
        }
        if self._camera.target_layer is not None:
            info["target_block"] = {
                "layer": self._camera.target_layer,
                "color": self._camera.target_color,
            }
        return info

    def _normalized_score(self) -> float:
        return round(self._raw_points / PERFECT_RAW_SCORE * 100.0, 2)

    @staticmethod
    def _format_score(score: float) -> str:
        return f"{score:.2f}"

    @staticmethod
    def _validate_change_viewpoint(action: Any) -> str | None:
        if not isinstance(action, dict):
            return "action must be a JSON object"
        if action.get("type") != "ChangeViewpoint":
            return "type must be ChangeViewpoint, Push, or PlaceBack"
        required = ("direction", "elevation_layer", "distance")
        missing = [field for field in required if field not in action]
        if missing:
            return f"missing field(s): {', '.join(missing)}"
        if action["direction"] not in DIRECTIONS:
            return f"direction must be one of {', '.join(DIRECTIONS)}"
        elev = action["elevation_layer"]
        if not isinstance(elev, int) or not 1 <= elev <= 18:
            return "elevation_layer must be an integer from 1 to 18"
        if action["distance"] not in DISTANCES:
            return f"distance must be one of {', '.join(DISTANCES)}"
        target_block = action.get("target_block")
        if target_block is not None:
            if not isinstance(target_block, dict):
                return "target_block must be an object with layer and color"
            if "layer" not in target_block or "color" not in target_block:
                return "target_block must have layer and color fields"
            if target_block["color"] not in ("Blue", "Brown", "Red"):
                return "target_block color must be Blue, Brown, or Red"
        return None

    def _last_outcome(self) -> str:
        if self._simulation is None or not self._simulation.last_frames:
            return ""
        return str(self._simulation.last_frames[-1]["phase"])

    def _tower_state(self) -> list[dict[str, Any]]:
        if self._simulation is None:
            return []
        colors = {block.spec.internal_id: block.spec.rgb for block in self._simulation.blocks}
        return [
            {
                "id": internal_id,
                "position": position,
                "rotation": rotation,
                "color": colors[internal_id],
            }
            for internal_id, position, rotation in self._simulation.transforms()
        ]

    def _available_placement_positions(self) -> tuple[str, ...]:
        if self._simulation is None:
            return ()
        return self._simulation.available_placement_positions

    def _render_png(self) -> bytes:
        if self._simulation is None:
            raise RuntimeError("Call reset() before rendering")
        from jenga.render import CameraPose, render_png

        pose = CameraPose.from_viewpoint(
            direction=self._camera.direction,
            elevation_layer=self._camera.elevation_layer,
            distance_cm=DISTANCES[self._camera.distance],
        )
        target = self._resolve_target()
        return render_png(self._simulation, pose, target=target)

    def _resolve_target(self) -> tuple[float, float, float] | None:
        if self._simulation is None or self._camera.target_layer is None:
            return None
        from jenga.render import CameraPose
        import pybullet as bullet

        for block in self._simulation.blocks:
            if (block.spec.layer == self._camera.target_layer
                    and block.spec.color_name == self._camera.target_color
                    and block.body_id not in self._simulation.retired_body_ids):
                pos, _ = bullet.getBasePositionAndOrientation(
                    block.body_id, physicsClientId=self._simulation.client_id
                )
                return tuple(pos)
        return CameraPose.target_for_layer(self._camera.elevation_layer)
