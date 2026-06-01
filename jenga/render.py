"""Deterministic TinyRenderer camera output encoded as PNG."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import pybullet as bullet  # pyright: ignore[reportMissingImports]

from jenga.settings import DEFAULT_SETTINGS
from jenga.sim import JengaSimulation

RENDER = DEFAULT_SETTINGS.render
IMAGE_WIDTH = RENDER.image_width
IMAGE_HEIGHT = RENDER.image_height
TOWER_MIDPOINT = RENDER.tower_midpoint


DIRECTION_AZIMUTHS = {
    "N": 0, "NE": 45, "E": 90, "SE": 135,
    "S": 180, "SW": 225, "W": 270, "NW": 315,
}

BLOCK_HEIGHT = DEFAULT_SETTINGS.geometry.block_height


@dataclass(frozen=True)
class CameraPose:
    azimuth: float
    pitch: float
    distance_cm: float
    elevation_layer: int = 9

    @classmethod
    def from_viewpoint(
        cls,
        direction: str,
        elevation_layer: int,
        distance_cm: float,
    ) -> CameraPose:
        azimuth = DIRECTION_AZIMUTHS.get(direction, 225)
        pitch = 15.0
        return cls(
            azimuth=azimuth,
            pitch=pitch,
            distance_cm=distance_cm,
            elevation_layer=elevation_layer,
        )

    @staticmethod
    def target_for_layer(elevation_layer: int) -> tuple[float, float, float]:
        return (0.0, 0.0, (elevation_layer - 0.5) * BLOCK_HEIGHT)

    def position(self) -> tuple[float, float, float]:
        import math

        yaw = math.radians(self.azimuth)
        distance = self.distance_cm / 100.0
        anchor_x, anchor_y, anchor_z = self.target_for_layer(self.elevation_layer)
        return (
            anchor_x + math.sin(yaw) * distance,
            anchor_y - math.cos(yaw) * distance,
            anchor_z,
        )


def render_png(
    simulation: JengaSimulation,
    camera: CameraPose,
    target: tuple[float, float, float] | None = None,
) -> bytes:
    default_target = CameraPose.target_for_layer(camera.elevation_layer)
    aim = target if target is not None else default_target
    eye = camera.position()
    view = bullet.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=aim,
        cameraUpVector=(0.0, 0.0, 1.0),
    )
    projection = bullet.computeProjectionMatrixFOV(
        fov=RENDER.field_of_view_degrees,
        aspect=1.0,
        nearVal=RENDER.near_plane,
        farVal=RENDER.far_plane,
    )
    _, _, rgba, _, _ = bullet.getCameraImage(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=bullet.ER_TINY_RENDERER,
        shadow=1,
        lightDirection=RENDER.light_direction,
        lightColor=RENDER.light_color,
        lightAmbientCoeff=RENDER.light_ambient_coefficient,
        lightDiffuseCoeff=RENDER.light_diffuse_coefficient,
        lightSpecularCoeff=RENDER.light_specular_coefficient,
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
