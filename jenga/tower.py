"""Tower constants and deterministic prebuilt geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

BLOCK_LENGTH = 0.075
BLOCK_WIDTH = 0.025
BLOCK_HEIGHT = 0.015
BLOCK_MASS = 0.120
BLOCK_CLEARANCE = 0.0002
LAYER_COUNT = 18
BLOCKS_PER_LAYER = 3

BASE_SIZE = (0.25, 0.25, 0.045)
BASE_CENTER_Z = -BASE_SIZE[2] / 2
FLOOR_Z = -BASE_SIZE[2]
FLOOR_SIZE = (4.0, 4.0, 0.01)
FLOOR_CENTER_Z = FLOOR_Z - FLOOR_SIZE[2] / 2

LATERAL_FRICTION = 0.40
ROLLING_FRICTION = 0.0
SPINNING_FRICTION = 0.0
LINEAR_DAMPING = 0.04
ANGULAR_DAMPING = 0.04


class Orientation(str, Enum):
    NORTH_SOUTH = "north-south"
    EAST_WEST = "east-west"


@dataclass(frozen=True)
class SlotDefinition:
    name: str
    offset: float
    color_name: str
    rgb: tuple[int, int, int]

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return (*(channel / 255 for channel in self.rgb), 1.0)


NORTH_SOUTH_SLOTS = (
    SlotDefinition("East", -(BLOCK_WIDTH + BLOCK_CLEARANCE), "Red", (160, 72, 72)),
    SlotDefinition("Middle", 0.0, "Lime", (120, 145, 70)),
    SlotDefinition("West", BLOCK_WIDTH + BLOCK_CLEARANCE, "Blue", (70, 100, 165)),
)
EAST_WEST_SLOTS = (
    SlotDefinition("South", -(BLOCK_WIDTH + BLOCK_CLEARANCE), "Wintergreen", (64, 148, 127)),
    SlotDefinition("Middle", 0.0, "Purple", (123, 84, 152)),
    SlotDefinition("North", BLOCK_WIDTH + BLOCK_CLEARANCE, "Brown", (124, 91, 63)),
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


def build_prebuilt_tower() -> tuple[BlockSpec, ...]:
    """Build the exact prebuilt tower used by every seed."""

    blocks: list[BlockSpec] = []
    lower_top = 0.0

    for layer_index in range(LAYER_COUNT):
        layer_number = layer_index + 1
        orientation = (
            Orientation.NORTH_SOUTH if layer_index % 2 == 0 else Orientation.EAST_WEST
        )
        slots = NORTH_SOUTH_SLOTS if orientation == Orientation.NORTH_SOUTH else EAST_WEST_SLOTS
        dimensions = (BLOCK_LENGTH, BLOCK_WIDTH, BLOCK_HEIGHT)
        for slot_index, slot in enumerate(slots):
            if orientation == Orientation.NORTH_SOUTH:
                x = slot.offset
                y = 0.0
            else:
                x = 0.0
                y = slot.offset

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
                    yaw_degrees=0.0,
                )
            )
        lower_top += BLOCK_HEIGHT

    return tuple(blocks)
