"""JengaBench environment with deterministic static-tower observations."""

from __future__ import annotations

import json
import math
import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bench_common.env_sdk.base import BaseEnv, StepResult

if TYPE_CHECKING:
    from jenga.sim import JengaSimulation

PNG_CONTENT_TYPE = "image/png"
PERFECT_RAW_SCORE = 98.0
EXTRACTION_COUNTDOWN_START = 10
CONTEXT_HISTORY_LIMIT = 5
INVALID_ACTION_PENALTY = -0.5
EXTRACTION_TIMEOUT_PENALTY = -1.0

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
        self._context_history: list[str] = []
        self._latest_context = ""
        self._moves_until_extraction = EXTRACTION_COUNTDOWN_START
        self._raw_points = 0.0
        self._successful_extractions = 0
        self._terminated = False
        self._termination_reason = ""
        self._seed: int | None = None
        self._simulation: JengaSimulation | None = None
        self._episode_replay: dict[str, Any] = {"schema_version": "1", "initial_frame": None, "steps": []}
        self._active_action: Any = None
        self._active_agent_frame: str | None = None
        self._active_physics_frames: list[dict[str, Any]] = []

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        del params
        from jenga.sim import JengaSimulation

        self.close()
        self._camera = CameraState()
        self._context_history = []
        self._latest_context = ""
        self._moves_until_extraction = EXTRACTION_COUNTDOWN_START
        self._raw_points = 0.0
        self._successful_extractions = 0
        self._terminated = False
        self._termination_reason = ""
        self._seed = seed
        self._simulation = JengaSimulation()
        try:
            self._simulation.reset(seed)
        except Exception:
            self.close()
            raise
        self._episode_replay = {
            "schema_version": "1",
            "initial_frame": self._simulation.last_frames[0] if self._simulation.last_frames else None,
            "steps": [],
        }
        self._active_action = None
        self._active_agent_frame = None
        self._active_physics_frames = []
        return {
            "data": self._render_png(),
            "content_type": PNG_CONTENT_TYPE,
            "system_prompt": self._system_prompt(),
        }

    def step(self, action: Any) -> StepResult:
        self._active_action = action
        self._active_agent_frame = base64.b64encode(self._render_png()).decode("ascii")
        self._active_physics_frames = []
        if self._terminated:
            return self._result(
                reward=INVALID_ACTION_PENALTY,
                event="invalid_action: episode already terminated",
            )

        context, error = self._validate_action_context(action)
        if error is not None:
            self._raw_points += INVALID_ACTION_PENALTY
            return self._result_after_non_extraction_step(
                reward=INVALID_ACTION_PENALTY,
                events=[f"invalid_action: {error}"],
            )

        assert context is not None
        self._remember_context(context)

        if action.get("type") == "Push":
            return self._step_push(action)
        if action.get("type") == "PlaceBack":
            return self._step_place_back(action)

        error = self._validate_change_viewpoint(action)
        if error is not None:
            self._raw_points += INVALID_ACTION_PENALTY
            return self._result_after_non_extraction_step(
                reward=INVALID_ACTION_PENALTY,
                events=[f"invalid_action: {error}"],
            )

        target_block = action.get("target_block")
        self._camera = CameraState(
            direction=action["direction"],
            elevation_layer=int(action["elevation_layer"]),
            distance=action["distance"],
            target_layer=int(target_block["layer"]) if target_block else None,
            target_color=target_block.get("color") if target_block else None,
        )
        return self._result_after_non_extraction_step(
            reward=0.0,
            events=["viewpoint_changed"],
        )

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
            self._active_physics_frames = list(push_result.frames)
        except PushValidationError as exc:
            self._raw_points += INVALID_ACTION_PENALTY
            return self._result_after_non_extraction_step(
                reward=INVALID_ACTION_PENALTY,
                events=[f"invalid_action: {exc}"],
            )

        reward = 0.0
        if push_result.outcome == "extracted":
            reward = 1.0
            self._raw_points += reward
            self._successful_extractions += 1
            self._moves_until_extraction = EXTRACTION_COUNTDOWN_START
            self._terminated = self._successful_extractions >= PERFECT_RAW_SCORE
            self._termination_reason = "perfect_completion" if self._terminated else ""
            return self._result(
                reward=reward,
                event="push_extracted",
            )

        self._terminated = push_result.outcome == "collapse"
        self._termination_reason = push_result.outcome if self._terminated else ""
        return self._result_after_non_extraction_step(
            reward=0.0,
            events=[f"push_{push_result.outcome}"],
        )

    def _step_place_back(self, action: dict[str, Any]) -> StepResult:
        if self._simulation is None:
            raise RuntimeError("Call reset() before stepping")
        from jenga.sim import PlaceRequest, PlaceValidationError

        try:
            request = PlaceRequest(
                position=action.get("position"),
            )
            place_result = self._simulation.place_back(request)
            self._active_physics_frames = list(place_result.frames)
        except PlaceValidationError as exc:
            self._raw_points += INVALID_ACTION_PENALTY
            return self._result_after_non_extraction_step(
                reward=INVALID_ACTION_PENALTY,
                events=[f"invalid_action: {exc}"],
            )

        self._terminated = place_result.outcome == "collapse"
        self._termination_reason = place_result.outcome if self._terminated else ""
        return self._result_after_non_extraction_step(
            reward=0.0,
            events=[f"place_back_{place_result.outcome}"],
        )

    def render(self, mode: str = "rgb_array") -> Any:
        if mode in ("rgb_array", "png"):
            return self._render_png()
        return self._camera_info()

    def close(self) -> None:
        if self._simulation is not None:
            self._simulation.close()
            self._simulation = None

    def _result_after_non_extraction_step(self, *, reward: float, events: list[str]) -> StepResult:
        self._moves_until_extraction -= 1
        if self._moves_until_extraction <= 0:
            reward += EXTRACTION_TIMEOUT_PENALTY
            self._raw_points += EXTRACTION_TIMEOUT_PENALTY
            self._moves_until_extraction = 0
            self._terminated = True
            self._termination_reason = "extraction_timeout"
            events = [*events, "extraction_timeout"]
        return self._result(reward=reward, event=events)

    def _result(self, *, reward: float, event: str | list[str]) -> StepResult:
        events = [event] if isinstance(event, str) else event
        self._record_replay_step(reward=reward, events=events)
        info = {
            "raw_points": self._format_score(self._raw_points),
            "normalized_score": self._format_score(self._normalized_score()),
            "blocks_removed": str(self._successful_extractions),
            "phase": self._simulation.phase if self._simulation is not None else "",
            "available_placement_positions": json.dumps(self._available_placement_positions()),
            "camera_state": json.dumps(self._camera_info(), sort_keys=True),
            "events": json.dumps(events),
            "termination_reason": self._termination_reason,
            "moves_until_extraction": str(self._moves_until_extraction),
            "latest_context": self._latest_context,
            "recent_contexts": json.dumps(self._context_history),
            "replay_schema_version": "1",
            "episode_replay": json.dumps(self._episode_replay, sort_keys=True),
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

    def _record_replay_step(self, *, reward: float, events: list[str]) -> None:
        action = self._active_action if isinstance(self._active_action, dict) else {}
        self._episode_replay["steps"].append(
            {
                "step": len(self._episode_replay["steps"]) + 1,
                "action": action,
                "see": action.get("see", "") if isinstance(action, dict) else "",
                "do": action.get("do", "") if isinstance(action, dict) else "",
                "next": action.get("next", "") if isinstance(action, dict) else "",
                "reward": reward,
                "terminated": self._terminated,
                "truncated": False,
                "events": events,
                "camera_state": self._camera_info(),
                "agent_frame": self._active_agent_frame,
                "tower_state": self._tower_state(),
                "physics_frames": self._active_physics_frames,
            }
        )

    def _system_prompt(self) -> str:
        context_history = (
            " | ".join(f"{index + 1}. {context}" for index, context in enumerate(self._context_history))
            if self._context_history
            else "none"
        )
        return (
            "You are playing JengaBench. Use the image as the current camera view. "
            'Return exactly one flat JSON action object. Every action must include three reasoning fields: '
            '"see" (describe what you observe in the image), '
            '"do" (explain why you chose this action), '
            '"next" (state your plan after this move). '
            "The action must be one of "
            '{"type":"ChangeViewpoint","see":"...","do":"...","next":"...","direction":"N|NE|E|SE|S|SW|W|NW",'
            '"elevation_layer":1..18,"distance":"Close|Medium|Full",'
            '"target_block":{"layer":int,"color":"Blue|Green|Red"}} or '
            '{"type":"Push","see":"...","do":"...","next":"...","layer":"1..one below current top layer","color":"Blue|Green|Red",'
            '"face":"North|South (odd layers) or East|West (even layers)","contact":"center|left|right",'
            '"intensity":"Gentle (barely moves block)|Firm (strong push)|Hard (full force, may topple tower)"} or '
            '{"type":"PlaceBack","see":"...","do":"...","next":"...","position":"<one of available_placement_positions>"}. '
            f"Camera: direction={self._camera.direction}, "
            f"elevation_layer={self._camera.elevation_layer}, "
            f"distance={self._camera.distance}. "
            f"Phase: {self._simulation.phase if self._simulation is not None else 'push'}. "
            f"Successful extractions: {self._successful_extractions}. "
            f"Moves remaining before the next successful extraction must happen: {self._moves_until_extraction}. "
            f"Push layer range: 1..{self._simulation.max_push_layer if self._simulation is not None else 17}. "
            "Block orientation: odd layers (1,3,5,...) run North-South so push faces are North or South; "
            "even layers (2,4,6,...) run East-West so push faces are East or West. "
            f"Available placement positions: {', '.join(self._available_placement_positions()) or 'none'}. "
            f"Last 5 action logs (oldest to newest): {context_history}. "
            "Only a fully extracted block resets the move countdown to 10; changing viewpoint, placing back, "
            "invalid actions, or pushes that do not extract a block still consume a move. "
            "If the countdown reaches 0, the episode terminates with a -1 point penalty. "
            "After a block is extracted from a layer, that block no longer exists there — do not try to push it again. "
            "Use Firm or Hard intensity to extract blocks; Gentle rarely pushes a block out. "
            "Minimize viewpoint changes — each one costs a move."
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

    def _remember_context(self, context: str) -> None:
        self._latest_context = context
        self._context_history.append(context)
        self._context_history = self._context_history[-CONTEXT_HISTORY_LIMIT:]

    @staticmethod
    def _validate_action_context(action: Any) -> tuple[str | None, str | None]:
        if not isinstance(action, dict):
            return None, "action must be a JSON object"
        see = action.get("see")
        do = action.get("do")
        nxt = action.get("next")
        if isinstance(see, str) and see.strip() and isinstance(do, str) and do.strip() and isinstance(nxt, str) and nxt.strip():
            return f"SEE: {see.strip()} | DO: {do.strip()} | NEXT: {nxt.strip()}", None
        context = action.get("context")
        if isinstance(context, str) and context.strip():
            return context, None
        return None, "action must include either see/do/next fields or a context string"

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
            if target_block["color"] not in ("Blue", "Green", "Red"):
                return "target_block color must be Blue, Green, or Red"
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
