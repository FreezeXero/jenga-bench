"""Tower definitions and deterministic seeded geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random

from jenga.settings import DEFAULT_SETTINGS, JengaSettings

GEOMETRY = DEFAULT_SETTINGS.geometry
BLOCK_LENGTH = GEOMETRY.block_length
BLOCK_WIDTH = GEOMETRY.block_width
BLOCK_HEIGHT = GEOMETRY.block_height
BLOCK_MASS = GEOMETRY.block_mass
BLOCK_CLEARANCE = GEOMETRY.block_clearance
LAYER_COUNT = GEOMETRY.layer_count
BLOCKS_PER_LAYER = GEOMETRY.blocks_per_layer

BASE_SIZE = GEOMETRY.base_size
BASE_CENTER_Z = -BASE_SIZE[2] / 2
FLOOR_Z = -BASE_SIZE[2]
FLOOR_SIZE = GEOMETRY.floor_size
FLOOR_CENTER_Z = FLOOR_Z - FLOOR_SIZE[2] / 2


class Orientation(str, Enum):
    NORTH_SOUTH = "north-south"
    EAST_WEST = "east-west"


@dataclass(frozen=True)
class SlotDefinition:
    name: str
    offset: float
    color_name: str
    rgb: tuple[int, int, int]


NORTH_SOUTH_SLOTS = (
    SlotDefinition("East", -(BLOCK_WIDTH + BLOCK_CLEARANCE), "Blue", (128, 153, 238)),
    SlotDefinition("Middle", 0.0, "Green", (74, 176, 120)),
    SlotDefinition("West", BLOCK_WIDTH + BLOCK_CLEARANCE, "Red", (238, 117, 99)),
)
EAST_WEST_SLOTS = (
    SlotDefinition("South", -(BLOCK_WIDTH + BLOCK_CLEARANCE), "Blue", (128, 153, 238)),
    SlotDefinition("Middle", 0.0, "Green", (74, 176, 120)),
    SlotDefinition("North", BLOCK_WIDTH + BLOCK_CLEARANCE, "Red", (238, 117, 99)),
)


@dataclass(frozen=True)
class BlockSpec:
    internal_id: str
    layer: int
    orientation: Orientation
    slot: str
    color_name: str
    rgb: tuple[int, int, int]
    dimensions: tuple[float, float, float]
    position: tuple[float, float, float]
    yaw_degrees: float

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return (*(channel / 255 for channel in self.rgb), 1.0)


def build_prebuilt_tower(
    seed: int | None = 0, settings: JengaSettings = DEFAULT_SETTINGS
) -> tuple[BlockSpec, ...]:
    """Build a repeatable tower variant from a restart seed."""

    geometry = settings.geometry
    randomness = settings.randomness
    rng = random.Random(0 if seed is None else seed)
    blocks: list[BlockSpec] = []
    lower_top = 0.0
    layer_x = 0.0
    layer_y = 0.0

    for layer_index in range(geometry.layer_count):
        layer_number = layer_index + 1
        if layer_index:
            layer_x += rng.uniform(-randomness.layer_shift_step, randomness.layer_shift_step)
            layer_y += rng.uniform(-randomness.layer_shift_step, randomness.layer_shift_step)
        orientation = (
            Orientation.NORTH_SOUTH if layer_index % 2 == 0 else Orientation.EAST_WEST
        )
        slots = NORTH_SOUTH_SLOTS if orientation == Orientation.NORTH_SOUTH else EAST_WEST_SLOTS
        dimensions = (geometry.block_length, geometry.block_width, geometry.block_height)
        extra_gap = rng.uniform(0.0, randomness.extra_layer_gap)
        yaw_degrees = rng.uniform(-randomness.layer_yaw_degrees, randomness.layer_yaw_degrees)
        for slot_index, slot in enumerate(slots):
            row_offset = (slot_index - 1) * (
                geometry.block_width + geometry.block_clearance + extra_gap
            )
            longitudinal_offset = rng.uniform(
                -randomness.block_longitudinal_offset,
                randomness.block_longitudinal_offset,
            )
            if orientation == Orientation.NORTH_SOUTH:
                x = layer_x + row_offset
                y = layer_y + longitudinal_offset
            else:
                x = layer_x + longitudinal_offset
                y = layer_y + row_offset

            blocks.append(
                BlockSpec(
                    internal_id=f"block-{layer_number:02d}-{slot_index}",
                    layer=layer_number,
                    orientation=orientation,
                    slot=slot.name,
                    color_name=slot.color_name,
                    rgb=slot.rgb,
                    dimensions=dimensions,
                    position=(x, y, lower_top + dimensions[2] / 2),
                    yaw_degrees=yaw_degrees,
                )
            )
        lower_top += geometry.block_height

    return tuple(blocks)
