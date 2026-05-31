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

@dataclass
class CameraState:
    azimuth: float = 225.0
    pitch: float = 15.0
    distance_cm: float = 45.0


class JengaBenchEnv(BaseEnv):
    """Image-producing environment backed by one authoritative PyBullet client."""

    def __init__(self) -> None:
        self._camera = CameraState()
        self._consecutive_viewpoints = 0
        self._raw_points = 0.0
        self._terminated = False
        self._seed: int | None = None
        self._simulation: JengaSimulation | None = None

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        del params
        from jenga.sim import JengaSimulation

        self.close()
        self._camera = CameraState()
        self._consecutive_viewpoints = 0
        self._raw_points = 0.0
        self._terminated = False
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
            return self._result(
                reward=INVALID_ACTION_PENALTY,
                event="invalid_action: episode already terminated",
            )

        error = self._validate_change_viewpoint(action)
        if error is not None:
            self._raw_points += INVALID_ACTION_PENALTY
            return self._result(
                reward=INVALID_ACTION_PENALTY,
                event=f"invalid_action: {error}",
            )

        self._camera = CameraState(
            azimuth=float(action["azimuth"]) % 360.0,
            pitch=float(action["pitch"]),
            distance_cm=float(action["distance_cm"]),
        )
        self._consecutive_viewpoints += 1

        reward = 0.0
        event = "viewpoint_changed"
        if self._consecutive_viewpoints >= VIEWPOINT_LIMIT:
            reward = VIEWPOINT_TIMEOUT_PENALTY
            self._raw_points += reward
            self._terminated = True
            event = "viewpoint_timeout"

        return self._result(reward=reward, event=event)

    def render(self, mode: str = "rgb_array") -> Any:
        if mode in ("rgb_array", "png"):
            return self._render_png()
        return self._camera_info()

    def close(self) -> None:
        if self._simulation is not None:
            self._simulation.close()
            self._simulation = None

    def _result(self, *, reward: float, event: str) -> StepResult:
        info = {
            "raw_points": self._format_score(self._raw_points),
            "normalized_score": self._format_score(self._normalized_score()),
            "blocks_removed": "0",
            "camera_state": json.dumps(self._camera_info(), sort_keys=True),
            "events": json.dumps([event]),
            "termination_reason": "viewpoint_timeout" if self._terminated else "",
        }
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
            "Only ChangeViewpoint is available in this contract slice. Return exactly one JSON "
            'action: {"type":"ChangeViewpoint","azimuth":0..360,"pitch":-90..90,'
            '"distance_cm":20..120}. '
            f"Camera: azimuth={self._camera.azimuth:.2f}, pitch={self._camera.pitch:.2f}, "
            f"distance_cm={self._camera.distance_cm:.2f}. "
            f"Consecutive viewpoints: {self._consecutive_viewpoints}/{VIEWPOINT_LIMIT}. "
            "The tenth consecutive viewpoint terminates the episode with a -10 point penalty."
        )

    def _camera_info(self) -> dict[str, float]:
        return {
            "azimuth": round(self._camera.azimuth, 2),
            "pitch": round(self._camera.pitch, 2),
            "distance_cm": round(self._camera.distance_cm, 2),
        }

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
            return "only ChangeViewpoint is implemented in the static-tower milestone"
        required = ("azimuth", "pitch", "distance_cm")
        missing = [field for field in required if field not in action]
        if missing:
            return f"missing field(s): {', '.join(missing)}"
        for field in required:
            value = action[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{field} must be numeric"
            if not math.isfinite(float(value)):
                return f"{field} must be finite"
        if not 0.0 <= float(action["azimuth"]) <= 360.0:
            return "azimuth must be between 0 and 360"
        if not -90.0 <= float(action["pitch"]) <= 90.0:
            return "pitch must be between -90 and 90"
        if not 20.0 <= float(action["distance_cm"]) <= 120.0:
            return "distance_cm must be between 20 and 120"
        return None

    def _render_png(self) -> bytes:
        if self._simulation is None:
            raise RuntimeError("Call reset() before rendering")
        from jenga.render import CameraPose, render_png

        return render_png(
            self._simulation,
            CameraPose(
                azimuth=self._camera.azimuth,
                pitch=self._camera.pitch,
                distance_cm=self._camera.distance_cm,
            ),
        )
