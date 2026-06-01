"""Local unscored browser inspector for the deterministic Jenga tower."""

from __future__ import annotations

import atexit
import asyncio
import logging
import math
import time
from pathlib import Path
from threading import Lock, RLock
from typing import Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jenga.render import DIRECTION_AZIMUTHS, CameraPose, render_png
from jenga.sim import (
    JengaSimulation,
    PlaceRequest,
    PlaceValidationError,
    PushRequest,
    PushValidationError,
)
from jenga.tower import BASE_CENTER_Z, BASE_SIZE, FLOOR_CENTER_Z, FLOOR_SIZE, Orientation

DEFAULT_CAMERA = CameraPose(azimuth=135.0, pitch=15.0, distance_cm=45.0)
MIN_INSPECTOR_PITCH = -45.0
MAX_INSPECTOR_PITCH = 75.0
STATIC_DIR = Path(__file__).with_name("static")


from typing import Literal, Optional

DISTANCES = {"Close": 15.0, "Medium": 30.0, "Full": 45.0}

class TargetBlock(BaseModel):
    layer: int
    color: Literal["Blue", "Green", "Red"]

class CameraRequest(BaseModel):
    direction: Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"] = "SW"
    elevation_layer: int = 9
    distance: Literal["Close", "Medium", "Full"] = "Full"
    target_block: Optional[TargetBlock] = None


class PreviewState:
    """Owns an inspector-only simulation and camera."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._simulation: JengaSimulation | None = None
        self._camera = DEFAULT_CAMERA
        self._push_lock = Lock()
        self._seed = 0
        self._terminated = False

    @property
    def camera(self) -> CameraPose:
        with self._lock:
            return self._camera

    def transforms(self) -> tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]:
        with self._lock:
            self._ensure_simulation()
            assert self._simulation is not None
            return self._simulation.transforms()

    def reset(self, seed: int = 0) -> tuple[bytes, CameraPose]:
        with self._lock:
            self._reset_locked(seed)
            return self._render_locked(), self._camera

    def reset_scene(self, seed: int = 0) -> dict[str, object]:
        with self._lock:
            self._reset_locked(seed)
            return self.scene()

    def frame(self, camera: CameraPose, target: tuple[float, float, float] | None = None) -> tuple[bytes, CameraPose]:
        with self._lock:
            self._ensure_simulation()
            self._camera = camera
            self._target = target
            return self._render_locked(), self._camera

    def scene(self) -> dict[str, object]:
        with self._lock:
            self._ensure_simulation()
            assert self._simulation is not None
            blocks = []
            transforms = {
                internal_id: (position, rotation)
                for internal_id, position, rotation in self._simulation.transforms()
            }
            for block in self._simulation.blocks:
                if block.body_id in self._simulation.retired_body_ids:
                    continue
                position, rotation = transforms[block.spec.internal_id]
                length, width, height = block.spec.dimensions
                size = (
                    (width, length, height)
                    if block.spec.orientation == Orientation.NORTH_SOUTH
                    else (length, width, height)
                )
                blocks.append(
                    {
                        "id": block.spec.internal_id,
                        "layer": block.spec.layer,
                        "slot": block.spec.slot,
                        "color_name": block.spec.color_name,
                        "position": position,
                        "rotation": rotation,
                        "size": size,
                        "color": block.spec.rgb,
                    }
                )
            return {
                "seed": self._seed,
                "camera": _camera_payload(self._camera),
                "target": (0.0, 0.0, 0.135),
                "floor": {
                    "position": (0.0, 0.0, FLOOR_CENTER_Z),
                    "size": FLOOR_SIZE,
                    "color": (150, 99, 66),
                },
                "base": {
                    "position": (0.0, 0.0, BASE_CENTER_Z),
                    "size": BASE_SIZE,
                    "color": (89, 56, 36),
                },
                "blocks": blocks,
                "phase": self._simulation.phase,
                "available_placement_positions": self._simulation.available_placement_positions,
                "top_layer": self._simulation.top_layer,
                "max_push_layer": self._simulation.max_push_layer,
            }

    def push(
        self,
        request: PushRequest,
        frame_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> tuple[dict[str, object], ...]:
        if not self._push_lock.acquire(blocking=False):
            raise RuntimeError("busy")
        try:
            with self._lock:
                if self._terminated:
                    raise RuntimeError("tower collapsed; reset is required")
                self._ensure_simulation()
                assert self._simulation is not None
                frames = self._simulation.push(
                    request,
                    frame_callback=frame_callback,
                    continue_after_collapse=True,
                ).frames
                self._terminated = frames[-1]["phase"] == "collapse"
                return frames
        finally:
            self._push_lock.release()

    def place_back(
        self,
        request: PlaceRequest,
        frame_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> tuple[dict[str, object], ...]:
        if not self._push_lock.acquire(blocking=False):
            raise RuntimeError("busy")
        try:
            with self._lock:
                if self._terminated:
                    raise RuntimeError("tower collapsed; reset is required")
                self._ensure_simulation()
                assert self._simulation is not None
                frames = self._simulation.place_back(
                    request,
                    frame_callback=frame_callback,
                    continue_after_collapse=True,
                ).frames
                self._terminated = frames[-1]["phase"] == "collapse"
                return frames
        finally:
            self._push_lock.release()

    def close(self) -> None:
        with self._lock:
            if self._simulation is not None:
                self._simulation.close()
                self._simulation = None

    def _ensure_simulation(self) -> None:
        if self._simulation is None:
            self._simulation = JengaSimulation(settings=_flat_to_settings(_current_settings))
            self._simulation.reset(seed=self._seed)

    def _reset_locked(self, seed: int) -> None:
        self.close()
        self._seed = seed
        self._simulation = JengaSimulation(settings=_flat_to_settings(_current_settings))
        self._simulation.reset(seed=seed)
        self._camera = DEFAULT_CAMERA
        self._terminated = False

    def _render_locked(self) -> bytes:
        assert self._simulation is not None
        target = getattr(self, "_target", None)
        return render_png(self._simulation, self._camera, target=target)

from jenga.settings import (
    DEFAULT_SETTINGS, JengaSettings,
    TowerGeometrySettings, TowerRandomnessSettings,
    PhysicsSettings, RenderSettings,
)

def _settings_to_flat(s: JengaSettings) -> dict:
    return {
        "block_length": s.geometry.block_length,
        "block_width": s.geometry.block_width,
        "block_height": s.geometry.block_height,
        "block_mass": s.geometry.block_mass,
        "layer_count": s.geometry.layer_count,
        "blocks_per_layer": s.geometry.blocks_per_layer,
        "block_longitudinal_offset": s.randomness.block_longitudinal_offset,
        "layer_shift_step": s.randomness.layer_shift_step,
        "layer_yaw_degrees": s.randomness.layer_yaw_degrees,
        "extra_layer_gap": s.randomness.extra_layer_gap,
        "gravity_z": s.physics.gravity[2],
        "lateral_friction": s.physics.lateral_friction,
        "restitution": s.physics.restitution,
        "push_force_multiplier": s.physics.push_force_multiplier,
        "intensity_gentle": dict(s.physics.intensities).get("Gentle", 0.05),
        "intensity_firm": dict(s.physics.intensities).get("Firm", 0.15),
        "intensity_hard": dict(s.physics.intensities).get("Hard", 0.40),
        "settle_timeout_seconds": s.physics.settle_timeout_seconds,
        "image_width": s.render.image_width,
        "image_height": s.render.image_height,
        "light_ambient_coefficient": s.render.light_ambient_coefficient,
        "light_diffuse_coefficient": s.render.light_diffuse_coefficient,
    }

def _flat_to_settings(flat: dict) -> JengaSettings:
    d = DEFAULT_SETTINGS
    return JengaSettings(
        geometry=TowerGeometrySettings(
            block_length=flat.get("block_length", d.geometry.block_length),
            block_width=flat.get("block_width", d.geometry.block_width),
            block_height=flat.get("block_height", d.geometry.block_height),
            block_mass=flat.get("block_mass", d.geometry.block_mass),
            layer_count=int(flat.get("layer_count", d.geometry.layer_count)),
            blocks_per_layer=int(flat.get("blocks_per_layer", d.geometry.blocks_per_layer)),
        ),
        randomness=TowerRandomnessSettings(
            block_longitudinal_offset=flat.get("block_longitudinal_offset", d.randomness.block_longitudinal_offset),
            layer_shift_step=flat.get("layer_shift_step", d.randomness.layer_shift_step),
            layer_yaw_degrees=flat.get("layer_yaw_degrees", d.randomness.layer_yaw_degrees),
            extra_layer_gap=flat.get("extra_layer_gap", d.randomness.extra_layer_gap),
        ),
        physics=PhysicsSettings(
            gravity=(0.0, 0.0, flat.get("gravity_z", d.physics.gravity[2])),
            lateral_friction=flat.get("lateral_friction", d.physics.lateral_friction),
            restitution=flat.get("restitution", d.physics.restitution),
            push_force_multiplier=flat.get("push_force_multiplier", d.physics.push_force_multiplier),
            intensities=(
                ("Gentle", flat.get("intensity_gentle", 0.05)),
                ("Firm", flat.get("intensity_firm", 0.15)),
                ("Hard", flat.get("intensity_hard", 0.40)),
            ),
            settle_timeout_seconds=flat.get("settle_timeout_seconds", d.physics.settle_timeout_seconds),
        ),
        render=RenderSettings(
            image_width=int(flat.get("image_width", d.render.image_width)),
            image_height=int(flat.get("image_height", d.render.image_height)),
            light_ambient_coefficient=flat.get("light_ambient_coefficient", d.render.light_ambient_coefficient),
            light_diffuse_coefficient=flat.get("light_diffuse_coefficient", d.render.light_diffuse_coefficient),
        ),
    )

_current_settings: dict = _settings_to_flat(DEFAULT_SETTINGS)
preview = PreviewState()
motion_lock = Lock()
atexit.register(preview.close)

app = FastAPI(title="JengaBench Live Tower Inspector", version="1.0.0")

@app.get("/api/settings")
def get_settings() -> dict:
    return _current_settings


@app.post("/api/settings")
async def update_settings(request: Request) -> dict:
    global _current_settings
    body = await request.json()
    _current_settings = {**_current_settings, **body}
    await asyncio.to_thread(preview.reset_scene, preview._seed)
    return _current_settings
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def disable_browser_cache(request: Request, call_next) -> Response:
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _png_response(data: bytes, camera: CameraPose) -> Response:
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Camera-Azimuth": f"{camera.azimuth:.2f}",
            "X-Camera-Pitch": f"{camera.pitch:.2f}",
            "X-Camera-Distance-Cm": f"{camera.distance_cm:.2f}",
        },
    )


def _camera_payload(camera: CameraPose) -> dict[str, float]:
    return {
        "azimuth": camera.azimuth,
        "pitch": camera.pitch,
        "distance_cm": camera.distance_cm,
    }


def _validated_camera(request: CameraRequest) -> tuple[CameraPose, tuple[float, float, float] | None]:
    if not 1 <= request.elevation_layer <= 18:
        raise HTTPException(status_code=422, detail="elevation_layer must be between 1 and 18")
    if request.target_block and not 1 <= request.target_block.layer <= 18:
        raise HTTPException(status_code=422, detail="target_block.layer must be between 1 and 18")
    pose = CameraPose.from_viewpoint(
        direction=request.direction,
        elevation_layer=request.elevation_layer,
        distance_cm=DISTANCES[request.distance],
    )
    target: tuple[float, float, float] | None = None
    if request.target_block:
        target = _resolve_target_block(request.target_block.layer, request.target_block.color)
    else:
        target = CameraPose.target_for_layer(request.elevation_layer)
    return pose, target


def _resolve_target_block(layer: int, color: str) -> tuple[float, float, float]:
    if preview._simulation is not None:
        import pybullet as bullet
        for block in preview._simulation.blocks:
            if (block.spec.layer == layer
                    and block.spec.color_name == color
                    and block.body_id not in preview._simulation.retired_body_ids):
                pos, _ = bullet.getBasePositionAndOrientation(
                    block.body_id, physicsClientId=preview._simulation.client_id
                )
                return tuple(pos)
    return CameraPose.target_for_layer(layer)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "jenga-bench-inspector"}


@app.post("/api/reset")
def reset(seed: int = 0) -> dict[str, object]:
    return preview.reset_scene(seed)


@app.get("/api/state")
def state() -> dict[str, object]:
    return preview.scene()


@app.post("/api/frame")
def frame(request: CameraRequest) -> Response:
    pose, target = _validated_camera(request)
    data, camera = preview.frame(pose, target)
    return _png_response(data, camera)


@app.post("/api/capture")
def capture(request: CameraRequest) -> Response:
    pose, target = _validated_camera(request)
    data, camera = preview.frame(pose, target)
    return _png_response(data, camera)


@app.websocket("/ws/sandbox")
async def sandbox(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            command = await websocket.receive_json()
            if command.get("type") == "Reset":
                if motion_lock.locked():
                    await websocket.send_json({"type": "error", "message": "busy"})
                    continue
                scene = await asyncio.to_thread(preview.reset_scene, int(command.get("seed", 0)))
                await websocket.send_json({"type": "scene", "scene": scene})
                continue
            if command.get("type") not in ("Push", "PlaceBack"):
                await websocket.send_json({"type": "error", "message": "type must be Reset, Push, or PlaceBack"})
                continue
            if not motion_lock.acquire(blocking=False):
                await websocket.send_json({"type": "error", "message": "busy"})
                continue
            try:
                try:
                    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
                    loop = asyncio.get_running_loop()

                    def emit(frame_payload: dict[str, object]) -> None:
                        loop.call_soon_threadsafe(queue.put_nowait, frame_payload)
                        time.sleep(0.001)

                    if command.get("type") == "Push":
                        motion_task = asyncio.create_task(
                            asyncio.to_thread(
                                preview.push,
                                PushRequest(
                                    layer=command.get("layer"),
                                    color=command.get("color"),
                                    face=command.get("face"),
                                    contact=command.get("contact"),
                                    intensity=command.get("intensity"),
                                ),
                                emit,
                            )
                        )
                    else:
                        motion_task = asyncio.create_task(
                            asyncio.to_thread(
                                preview.place_back,
                                PlaceRequest(
                                    position=command.get("position"),
                                ),
                                emit,
                            )
                        )
                    while not motion_task.done() or not queue.empty():
                        try:
                            frame_payload = await asyncio.wait_for(queue.get(), timeout=0.05)
                        except asyncio.TimeoutError:
                            continue
                        await websocket.send_json(frame_payload)
                        await asyncio.sleep(1.0 / 30.0)
                    frames = await motion_task
                except (PlaceValidationError, PushValidationError, RuntimeError) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue
                await websocket.send_json(
                    {
                        "type": "result",
                        "outcome": frames[-1]["phase"],
                        "frame_count": len(frames),
                        "scene": preview.scene(),
                    }
                )
            finally:
                motion_lock.release()
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(name)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level=args.log_level)
