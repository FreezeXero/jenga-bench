"""Deterministic procedural face-aware wood textures for TinyRenderer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random
import struct
import zlib
from pathlib import Path

from jenga.settings import DEFAULT_SETTINGS

RENDER = DEFAULT_SETTINGS.render


@dataclass(frozen=True)
class WoodIdentity:
    block_brightness: float
    temperature: float
    mottle_seed: int
    spot_seed: int
    end_brightness: float
    end_phase: float
    end_slope: float
    end_curve: float
    end_rotations: tuple[float, float]


def stable_seed(seed: int | None, internal_id: str) -> int:
    """Return a stable 32-bit FNV-1a hash for one seeded block identity."""
    value = 2166136261
    for byte in f"{0 if seed is None else seed}:{internal_id}".encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def wood_variant(seed: int | None, internal_id: str) -> int:
    return stable_seed(seed, internal_id) % RENDER.wood_texture_variant_count


def wood_identity(seed: int | None, variant: int) -> WoodIdentity:
    rng = random.Random(stable_seed(seed, f"variant-{variant}"))
    low, high = RENDER.wood_block_brightness_range
    end_low, end_high = RENDER.wood_end_brightness_range
    temperature_low, temperature_high = RENDER.wood_temperature_range
    return WoodIdentity(
        block_brightness=rng.uniform(low, high),
        temperature=rng.uniform(temperature_low, temperature_high),
        mottle_seed=rng.randrange(0xFFFFFFFF),
        spot_seed=rng.randrange(0xFFFFFFFF),
        end_brightness=rng.uniform(end_low, end_high),
        end_phase=rng.random() * math.tau,
        end_slope=rng.uniform(-0.7, 0.7),
        end_curve=rng.uniform(0.6, 1.5),
        end_rotations=(rng.random() * math.tau, rng.random() * math.tau),
    )


def _noise(seed: int, x: int, y: int) -> float:
    value = (seed ^ (x * 374761393) ^ (y * 668265263)) & 0xFFFFFFFF
    value = ((value ^ (value >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((value ^ (value >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF


def _smooth_noise(seed: int, u: float, v: float, scale: float) -> float:
    x = u * scale
    y = v * scale
    x0 = math.floor(x)
    y0 = math.floor(y)
    fx = x - x0
    fy = y - y0
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    top = _noise(seed, x0, y0) * (1.0 - fx) + _noise(seed, x0 + 1, y0) * fx
    bottom = _noise(seed, x0, y0 + 1) * (1.0 - fx) + _noise(seed, x0 + 1, y0 + 1) * fx
    return (top * (1.0 - fy) + bottom * fy) * 2.0 - 1.0


def _granular_shade(u: float, v: float, identity: WoodIdentity) -> float:
    # Rotated layers avoid visible square cells and stay smooth under camera motion.
    mottle_u = u * 0.91 + v * 0.41
    mottle_v = -u * 0.41 + v * 0.91
    detail_u = u * 0.67 - v * 0.74 + 3.17
    detail_v = u * 0.74 + v * 0.67 - 1.93
    mottle = _smooth_noise(
        identity.mottle_seed, mottle_u, mottle_v, RENDER.wood_mottle_scale
    ) * RENDER.wood_mottle_contrast
    detail = _smooth_noise(
        identity.spot_seed, detail_u, detail_v, RENDER.wood_detail_scale
    ) * RENDER.wood_detail_contrast
    return 1.0 + mottle + detail


def _rotate_uv(u: float, v: float, angle: float) -> tuple[float, float]:
    centered_u = u - 0.5
    centered_v = v - 0.5
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        centered_u * cosine - centered_v * sine + 0.5,
        centered_u * sine + centered_v * cosine + 0.5,
    )


def _grain(face: str, u: float, v: float, identity: WoodIdentity) -> tuple[float, float]:
    if face == "long_side":
        return _granular_shade(u, v, identity), 1.0
    variant = 1 if face.endswith("_b") else 0
    variant_phase = variant * 1.731
    variant_u, variant_v = _rotate_uv(u, v, identity.end_rotations[variant])
    centered = variant_u - 0.5
    rings = (
        variant_v * math.tau * RENDER.wood_end_ring_frequency
        + centered * centered * math.tau * identity.end_curve * 3.0
    )
    wave = abs(math.sin(rings + identity.end_phase + variant_phase + centered * identity.end_slope))
    line = max(0.0, (wave - RENDER.wood_end_line_threshold) / (1.0 - RENDER.wood_end_line_threshold))
    return _granular_shade(variant_u, variant_v, identity) - line * RENDER.wood_end_line_contrast, identity.end_brightness


def _temperature_scale(temperature: float) -> tuple[float, float, float]:
    """Shift wood between pale and warm boards without overpowering its base tint."""
    return 1.0 + temperature * 0.45, 1.0, 1.0 - temperature


@lru_cache(maxsize=1024)
def wood_texture_png(seed: int | None, variant: int, rgb: tuple[int, int, int]) -> bytes:
    """Return an RGB PNG atlas with broad, edge, and end-grain columns."""
    identity = wood_identity(seed, variant)
    temperature_scale = _temperature_scale(identity.temperature)
    cell = RENDER.wood_texture_cell_size
    faces = ("long_side", "end_a", "end_b")
    width = cell * len(faces)
    rows = bytearray()
    for y in range(cell):
        rows.append(0)
        for x in range(width):
            face = faces[x // cell]
            u = (x % cell) / cell
            v = y / cell
            grain, face_brightness = _grain(face, u, v, identity)
            shade = identity.block_brightness * face_brightness * grain
            rows.extend(
                min(255, max(0, round(channel * shade * channel_scale)))
                for channel, channel_scale in zip(rgb, temperature_scale)
            )
    return _encode_rgb_png(width, cell, rows)


def write_block_texture(
    directory: Path,
    seed: int | None,
    variant: int,
    color_name: str,
    rgb: tuple[int, int, int],
) -> Path:
    path = directory / f"{0 if seed is None else seed}-variant-{variant}-{color_name.lower()}-wood.png"
    if not path.exists():
        path.write_bytes(wood_texture_png(seed, variant, rgb))
    return path


def write_block_obj(path: Path, long_axis: str) -> Path:
    """Write a cuboid OBJ with broad, edge, and end atlas UVs."""
    if long_axis not in ("x", "y"):
        raise ValueError("long_axis must be x or y")
    faces = (
        ((1, 2, 6, 5), "edge" if long_axis == "x" else "end", ((0, 0), (1, 0), (1, 1), (0, 1))),
        ((4, 8, 7, 3), "edge" if long_axis == "x" else "end", ((0, 0), (0, 1), (1, 1), (1, 0))),
        ((1, 5, 8, 4), "end" if long_axis == "x" else "edge", ((0, 0), (0, 1), (1, 1), (1, 0))),
        ((2, 3, 7, 6), "end" if long_axis == "x" else "edge", ((0, 0), (1, 0), (1, 1), (0, 1))),
        ((1, 4, 3, 2), "broad", ((0, 0), (0, 1), (1, 1), (1, 0))),
        ((5, 6, 7, 8), "broad", ((0, 0), (1, 0), (1, 1), (0, 1))),
    )
    regions = {"long_side": 0, "end_a": 1, "end_b": 2}
    texture_lines = []
    face_lines = []
    texture_index = 1
    variants = {"edge": 0, "end": 0}
    for vertices, face, coordinates in faces:
        if face in ("broad", "edge"):
            face = "long_side"
        elif face in variants:
            suffix = "a" if variants[face] == 0 else "b"
            variants[face] += 1
            face = f"{face}_{suffix}"
        region = regions[face]
        for u, v in coordinates:
            texture_lines.append(f"vt {(region + u) / 3:.6f} {v:.6f}")
        face_lines.append("f " + " ".join(
            f"{vertex}/{texture_index + offset}" for offset, vertex in enumerate(vertices)
        ))
        texture_index += 4
    path.write_text(
        """# unit cuboid: broad, edge, and end UV atlas columns
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
""" + "\n".join(texture_lines + face_lines) + "\n",
        encoding="ascii",
    )
    return path


def _encode_rgb_png(width: int, height: int, rows: bytes | bytearray) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + chunk(b"IEND", b"")
    )
