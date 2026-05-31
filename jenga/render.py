"""Deterministic TinyRenderer camera output encoded as PNG."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import pybullet as bullet

from jenga.sim import JengaSimulation

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
TOWER_MIDPOINT = (0.0, 0.0, 0.135)


@dataclass(frozen=True)
class CameraPose:
    azimuth: float
    pitch: float
    distance_cm: float


def render_png(simulation: JengaSimulation, camera: CameraPose) -> bytes:
    view = bullet.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=TOWER_MIDPOINT,
        distance=camera.distance_cm / 100.0,
        yaw=camera.azimuth,
        pitch=-camera.pitch,
        roll=0.0,
        upAxisIndex=2,
    )
    projection = bullet.computeProjectionMatrixFOV(
        fov=52.0,
        aspect=1.0,
        nearVal=0.02,
        farVal=3.0,
    )
    _, _, rgba, _, _ = bullet.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=bullet.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=(3.0, -4.0, 6.0),
        lightColor=(1.0, 1.0, 1.0),
        lightAmbientCoeff=0.7,
        lightDiffuseCoeff=0.6,
        lightSpecularCoeff=0.05,
        physicsClientId=simulation.client_id,
    )
    return encode_rgb_png(IMAGE_WIDTH, IMAGE_HEIGHT, rgba)


def encode_rgb_png(width: int, height: int, rgba: object) -> bytes:
    values = bytes(rgba)
    raw_rows = bytearray()
    stride = width * 4
    for y in range(height):
        raw_rows.append(0)
        row = values[y * stride : (y + 1) * stride]
        for offset in range(0, len(row), 4):
            raw_rows.extend(row[offset : offset + 3])

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=9))
        + chunk(b"IEND", b"")
    )
