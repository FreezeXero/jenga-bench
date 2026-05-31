"""Local unscored browser inspector for the deterministic Jenga tower."""

from __future__ import annotations

import atexit
import asyncio
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

from jenga.render import CameraPose, render_png
from jenga.sim import JengaSimulation, PushRequest, PushValidationError
from jenga.tower import BASE_CENTER_Z, BASE_SIZE, Orientation

DEFAULT_CAMERA = CameraPose(azimuth=225.0, pitch=15.0, distance_cm=45.0)
MIN_INSPECTOR_PITCH = 0.0
MAX_INSPECTOR_PITCH = 75.0
STATIC_DIR = Path(__file__).with_name("static")


class CameraRequest(BaseModel):
    azimuth: float
    pitch: float
    distance_cm: float


class PreviewState:
    """Owns an inspector-only simulation and camera."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._simulation: JengaSimulation | None = None
        self._camera = DEFAULT_CAMERA
        self._push_lock = Lock()
        self._seed = 0

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

    def frame(self, camera: CameraPose) -> tuple[bytes, CameraPose]:
        with self._lock:
            self._ensure_simulation()
            self._camera = camera
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
                "base": {
                    "position": (0.0, 0.0, BASE_CENTER_Z),
                    "size": BASE_SIZE,
                    "color": (8, 8, 9),
                },
                "blocks": blocks,
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
                self._ensure_simulation()
                assert self._simulation is not None
                return self._simulation.push(
                    request,
                    frame_callback=frame_callback,
                    continue_after_collapse=True,
                ).frames
        finally:
            self._push_lock.release()

    def close(self) -> None:
        with self._lock:
            if self._simulation is not None:
                self._simulation.close()
                self._simulation = None

    def _ensure_simulation(self) -> None:
        if self._simulation is None:
            self._simulation = JengaSimulation()
            self._simulation.reset(seed=self._seed)

    def _reset_locked(self, seed: int) -> None:
        self.close()
        self._seed = seed
        self._simulation = JengaSimulation()
        self._simulation.reset(seed=seed)
        self._camera = DEFAULT_CAMERA

    def _render_locked(self) -> bytes:
        assert self._simulation is not None
        return render_png(self._simulation, self._camera)


preview = PreviewState()
motion_lock = Lock()
atexit.register(preview.close)

app = FastAPI(title="JengaBench Live Tower Inspector", version="1.0.0")
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


def _validated_camera(request: CameraRequest) -> CameraPose:
    values = (request.azimuth, request.pitch, request.distance_cm)
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(status_code=422, detail="camera values must be finite")
    if not 0.0 <= request.azimuth <= 360.0:
        raise HTTPException(status_code=422, detail="azimuth must be between 0 and 360")
    if not MIN_INSPECTOR_PITCH <= request.pitch <= MAX_INSPECTOR_PITCH:
        raise HTTPException(status_code=422, detail="pitch must be between 0 and 75")
    if not 20.0 <= request.distance_cm <= 120.0:
        raise HTTPException(status_code=422, detail="distance_cm must be between 20 and 120")
    return CameraPose(
        azimuth=request.azimuth % 360.0,
        pitch=request.pitch,
        distance_cm=request.distance_cm,
    )


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
    data, camera = preview.frame(_validated_camera(request))
    return _png_response(data, camera)


@app.post("/api/capture")
def capture(request: CameraRequest) -> Response:
    data, camera = preview.frame(_validated_camera(request))
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
            if command.get("type") != "Push":
                await websocket.send_json({"type": "error", "message": "type must be Reset or Push"})
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

                    push_task = asyncio.create_task(
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
                    while not push_task.done() or not queue.empty():
                        try:
                            frame_payload = await asyncio.wait_for(queue.get(), timeout=0.05)
                        except asyncio.TimeoutError:
                            continue
                        await websocket.send_json(frame_payload)
                        await asyncio.sleep(1.0 / 30.0)
                    frames = await push_task
                except (PushValidationError, RuntimeError) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue
                await websocket.send_json(
                    {
                        "type": "result",
                        "outcome": frames[-1]["phase"],
                        "frame_count": len(frames),
                    }
                )
            finally:
                motion_lock.release()
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
