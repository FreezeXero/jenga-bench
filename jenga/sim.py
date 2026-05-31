"""Authoritative deterministic PyBullet simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import math
from typing import Any, Callable

import pybullet as bullet  # pyright: ignore[reportMissingImports]

from jenga.tower import (
    BASE_CENTER_Z,
    BASE_SIZE,
    BLOCK_HEIGHT,
    BLOCK_LENGTH,
    BLOCK_MASS,
    FLOOR_CENTER_Z,
    FLOOR_SIZE,
    EAST_WEST_SLOTS,
    NORTH_SOUTH_SLOTS,
    BlockSpec,
    Orientation,
    build_prebuilt_tower,
)
from jenga.settings import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)

PHYSICS = DEFAULT_SETTINGS.physics
TIMESTEP = PHYSICS.timestep
SOLVER_ITERATIONS = PHYSICS.solver_iterations
SETTLE_TIMEOUT_SECONDS = PHYSICS.settle_timeout_seconds
VIEWER_COLLAPSE_TAIL_TIMEOUT_SECONDS = PHYSICS.viewer_collapse_tail_timeout_seconds
VIEWER_COLLAPSE_LINEAR_VELOCITY_THRESHOLD = PHYSICS.viewer_collapse_linear_velocity_threshold
VIEWER_COLLAPSE_ANGULAR_VELOCITY_THRESHOLD = PHYSICS.viewer_collapse_angular_velocity_threshold
VIEWER_COLLAPSE_STABLE_STEPS = PHYSICS.viewer_collapse_stable_steps
SETTLE_STABLE_STEPS = PHYSICS.settle_stable_steps
LINEAR_VELOCITY_THRESHOLD = PHYSICS.linear_velocity_threshold
ANGULAR_VELOCITY_THRESHOLD = PHYSICS.angular_velocity_threshold
RAMP_DURATION_SECONDS = PHYSICS.ramp_duration_seconds
RAMP_STEPS = round(RAMP_DURATION_SECONDS / TIMESTEP)
LATERAL_FRICTION = PHYSICS.lateral_friction
FRAME_SAMPLE_STEPS = PHYSICS.frame_sample_steps
PLACEMENT_DROP_HEIGHT = PHYSICS.placement_drop_height

COLLAPSE_CONSECUTIVE_STEPS = 30

CONTACTS = ("center", "left", "right")
VELOCITY_CAPS = dict(PHYSICS.intensities)
PUSH_FORCE_MULTIPLIER = PHYSICS.push_force_multiplier
VALID_FACES = {
    Orientation.NORTH_SOUTH: ("North", "South"),
    Orientation.EAST_WEST: ("East", "West"),
}


class TowerStabilityError(RuntimeError):
    """Raised when the deterministic tower cannot settle."""


class PushValidationError(ValueError):
    """Raised when a push request is invalid."""


class PlaceValidationError(ValueError):
    """Raised when a placement request is invalid."""


@dataclass
class BlockBody:
    spec: BlockSpec
    body_id: int


@dataclass(frozen=True)
class PushRequest:
    layer: int
    color: str
    face: str
    contact: str
    intensity: str


@dataclass(frozen=True)
class PushResult:
    outcome: str
    frames: tuple[dict[str, Any], ...]
    target_id: str
    ramp_steps: int
    settle_steps: int


@dataclass(frozen=True)
class PlaceRequest:
    position: str


@dataclass(frozen=True)
class PlaceResult:
    outcome: str
    frames: tuple[dict[str, Any], ...]
    target_id: str
    settle_steps: int


class JengaSimulation:
    """Owns one PyBullet DIRECT client and its dynamic tower."""

    def __init__(self) -> None:
        self.client_id = bullet.connect(bullet.DIRECT)
        self.blocks: tuple[BlockBody, ...] = ()
        self.base_body_id: int | None = None
        self.floor_body_id: int | None = None
        self.settle_steps: int | None = None
        self.last_frames: tuple[dict[str, Any], ...] = ()
        self.retired_body_ids: set[int] = set()
        self.held_block: BlockBody | None = None
        self.placement_layer: int | None = None
        self.placement_anchor: tuple[float, float] | None = None
        self.placement_surface_z: float | None = None
        self._configure()

    @property
    def phase(self) -> str:
        return "place_back" if self.held_block is not None else "push"

    @property
    def top_layer(self) -> int:
        return max(
            block.spec.layer for block in self.blocks if block.body_id not in self.retired_body_ids
        )

    @property
    def max_push_layer(self) -> int:
        return self.top_layer - 1

    @property
    def available_placement_positions(self) -> tuple[str, ...]:
        if self.held_block is None:
            return ()
        return self._available_positions_for_layer(self._next_placement_layer())

    def _available_positions_for_layer(self, layer: int) -> tuple[str, ...]:
        occupied = {
            self._position_for_slot(block.spec.orientation, block.spec.slot)
            for block in self.blocks
            if block.body_id not in self.retired_body_ids and block.spec.layer == layer
        }
        return tuple(position for position in ("Left", "Middle", "Right") if position not in occupied)

    @property
    def is_connected(self) -> bool:
        return bool(bullet.isConnected(self.client_id))

    def close(self) -> None:
        if self.is_connected:
            bullet.disconnect(self.client_id)

    def reset(self, seed: int | None) -> None:
        self._clear_world()
        self._build_world(build_prebuilt_tower(seed))
        if not self._settle():
            raise TowerStabilityError("prebuilt tower failed to settle")
        self._zero_velocities()
        self.last_frames = (self.frame(sequence=0, sim_time=0.0, phase="initial"),)

    def transforms(self) -> tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]:
        values = []
        for block in self.blocks:
            if block.body_id in self.retired_body_ids:
                continue
            position, rotation = bullet.getBasePositionAndOrientation(
                block.body_id, physicsClientId=self.client_id
            )
            values.append((block.spec.internal_id, tuple(position), tuple(rotation)))
        return tuple(values)

    def frame(self, *, sequence: int, sim_time: float, phase: str) -> dict[str, Any]:
        return {
            "type": "frame",
            "sequence": sequence,
            "sim_time": round(sim_time, 6),
            "phase": phase,
            "blocks": [
                {
                    "id": block.spec.internal_id,
                    "position": position,
                    "rotation": rotation,
                    "color": block.spec.rgb,
                    "size": self._block_size(block.spec),
                }
                for block in self.blocks
                if block.body_id not in self.retired_body_ids
                for position, rotation in (
                    bullet.getBasePositionAndOrientation(
                        block.body_id, physicsClientId=self.client_id
                    ),
                )
            ],
            "available_placement_positions": self.available_placement_positions,
        }

    def push(
        self,
        request: PushRequest,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
        *,
        continue_after_collapse: bool = False,
    ) -> PushResult:
        if self.held_block is not None:
            raise PushValidationError("PlaceBack is required before another Push")
        target = self._validate_push(request)
        frames = [self.frame(sequence=0, sim_time=0.0, phase="initial")]
        self._emit_frame(frames[-1], frame_callback)
        sequence = 0
        simulated_steps = 0

        direction = self._force_direction_aligned(target, request.face)
        contact = self._world_contact_point(target, request)
        load = self._get_load(target.body_id)
        breakaway = load * LATERAL_FRICTION
        peak_force = breakaway * PUSH_FORCE_MULTIPLIER
        velocity_cap = VELOCITY_CAPS[request.intensity]
        logger.debug("push load=%.3fN breakaway=%.3fN peak=%.3fN cap=%.2fm/s", load, breakaway, peak_force, velocity_cap)
        contact_snapshot = self._snapshot_vertical_contacts(target.body_id)
        collapse_lost_steps: dict[int, int] = {}
        extracted = False
        for ramp_step in range(1, RAMP_STEPS + 1):
            speed = sum(v ** 2 for v in bullet.getBaseVelocity(target.body_id, physicsClientId=self.client_id)[0]) ** 0.5
            if speed >= velocity_cap:
                bullet.stepSimulation(physicsClientId=self.client_id)
                simulated_steps += 1
            else:
                multiplier = math.sin(math.pi * ramp_step / (RAMP_STEPS + 1))
                force = tuple(value * peak_force * multiplier for value in direction)
                bullet.applyExternalForce(
                    target.body_id,
                    -1,
                    forceObj=force,
                    posObj=contact,
                    flags=bullet.WORLD_FRAME,
                    physicsClientId=self.client_id,
                )
                bullet.stepSimulation(physicsClientId=self.client_id)
                simulated_steps += 1
            if ramp_step % FRAME_SAMPLE_STEPS == 0:
                sequence += 1
                frames.append(self.frame(sequence=sequence, sim_time=simulated_steps * TIMESTEP, phase="ramp"))
                self._emit_frame(frames[-1], frame_callback)
            extracted = extracted or self._is_extracted(target)
            if self._check_collapse(contact_snapshot, collapse_lost_steps, target.body_id):
                if continue_after_collapse:
                    sequence, simulated_steps = self._settle_collapse_tail(
                        frames, sequence, simulated_steps, frame_callback
                    )
                return self._finish_push(
                    "collapse", target, frames, sequence, simulated_steps, 0, RAMP_STEPS, frame_callback
                )

        settled, extracted, sequence, simulated_steps, settle_steps = self._settle_with_frames(
            frames,
            sequence,
            simulated_steps,
            target,
            extracted,
            contact_snapshot,
            collapse_lost_steps,
            frame_callback,
        )
        if not settled:
            logger.debug("failed to settle in time — checking final state")
            if self._check_collapse(contact_snapshot, collapse_lost_steps, target.body_id):
                outcome = "collapse"
            elif extracted:
                outcome = "extracted"
            else:
                outcome = "settled"
        elif extracted:
            outcome = "extracted"
        else:
            outcome = "settled"
        if outcome == "collapse" and continue_after_collapse:
            sequence, simulated_steps = self._settle_collapse_tail(
                frames, sequence, simulated_steps, frame_callback
            )
        return self._finish_push(
            outcome, target, frames, sequence, simulated_steps, settle_steps, RAMP_STEPS, frame_callback
        )

    def _finish_push(
        self,
        outcome: str,
        target: BlockBody,
        frames: list[dict[str, Any]],
        sequence: int,
        simulated_steps: int,
        settle_steps: int,
        ramp_steps: int = RAMP_STEPS,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> PushResult:
        if outcome == "extracted":
            self.retired_body_ids.add(target.body_id)
            self.held_block = target
        self._zero_velocities()
        final_sequence = sequence + 1
        final = self.frame(sequence=final_sequence, sim_time=simulated_steps * TIMESTEP, phase=outcome)
        frames.append(final)
        self._emit_frame(final, frame_callback)
        if outcome == "extracted":
            bullet.removeBody(target.body_id, physicsClientId=self.client_id)
        self.last_frames = tuple(frames)
        return PushResult(
            outcome=outcome,
            frames=self.last_frames,
            target_id=target.spec.internal_id,
            ramp_steps=RAMP_STEPS,
            settle_steps=settle_steps,
        )

    def _validate_push(self, request: PushRequest) -> BlockBody:
        if (
            isinstance(request.layer, bool)
            or not isinstance(request.layer, int)
            or not 1 <= request.layer <= self.max_push_layer
        ):
            raise PushValidationError(
                f"layer must be an integer between 1 and {self.max_push_layer}; "
                "the top layer cannot be pushed"
            )
        target = next(
            (
                block
                for block in self.blocks
                if block.body_id not in self.retired_body_ids
                and block.spec.layer == request.layer
                and block.spec.color_name == request.color
            ),
            None,
        )
        if target is None:
            raise PushValidationError("color does not identify a block in the requested layer")
        if request.face not in VALID_FACES[target.spec.orientation]:
            valid = ", ".join(VALID_FACES[target.spec.orientation])
            raise PushValidationError(f"face must be one of: {valid}")
        if request.contact not in CONTACTS:
            raise PushValidationError("contact must be center, left, or right")
        if request.intensity not in VELOCITY_CAPS:
            raise PushValidationError("intensity must be Gentle, Firm, or Hard")
        return target

    def place_back(
        self,
        request: PlaceRequest,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
        *,
        continue_after_collapse: bool = False,
    ) -> PlaceResult:
        target = self._validate_place(request)
        layer = self._next_placement_layer()
        orientation = self._orientation_for_layer(layer)
        slots = NORTH_SOUTH_SLOTS if orientation == Orientation.NORTH_SOUTH else EAST_WEST_SLOTS
        slot = slots[("Left", "Middle", "Right").index(request.position)]
        anchor_x, anchor_y = self._placement_row_anchor()
        top_z = self._placement_row_surface()
        if orientation == Orientation.NORTH_SOUTH:
            position = (anchor_x + slot.offset, anchor_y, top_z + BLOCK_HEIGHT / 2 + PLACEMENT_DROP_HEIGHT)
            base_yaw = 0.0
        else:
            position = (anchor_x, anchor_y + slot.offset, top_z + BLOCK_HEIGHT / 2 + PLACEMENT_DROP_HEIGHT)
            base_yaw = 0.0
        spec = replace(
            target.spec,
            layer=layer,
            orientation=orientation,
            slot=slot.name,
            color_name=slot.color_name,
            rgb=slot.rgb,
            position=position,
            yaw_degrees=base_yaw,
        )
        frames = [self.frame(sequence=0, sim_time=0.0, phase="initial")]
        self._emit_frame(frames[-1], frame_callback)
        self.retired_body_ids.remove(target.body_id)
        placed = self._create_block_body(spec)
        self.blocks = tuple(placed if block is target else block for block in self.blocks)
        self.held_block = None
        sequence = 1
        simulated_steps = 0
        spawn = self.frame(sequence=sequence, sim_time=0.0, phase="place-drop")
        frames.append(spawn)
        self._emit_frame(spawn, frame_callback)
        place_snapshot = self._snapshot_vertical_contacts(placed.body_id)
        collapse_lost_steps: dict[int, int] = {}
        settled = False
        stable_steps = 0
        timeout_steps = round(SETTLE_TIMEOUT_SECONDS / TIMESTEP)
        settle_steps = timeout_steps
        for settle_step in range(1, timeout_steps + 1):
            bullet.stepSimulation(physicsClientId=self.client_id)
            simulated_steps += 1
            if settle_step % FRAME_SAMPLE_STEPS == 0:
                sequence += 1
                frame = self.frame(
                    sequence=sequence,
                    sim_time=simulated_steps * TIMESTEP,
                    phase="place-settle",
                )
                frames.append(frame)
                self._emit_frame(frame, frame_callback)
            if self._check_collapse(place_snapshot, collapse_lost_steps):
                break
            if self._all_blocks_below_velocity_thresholds():
                stable_steps += 1
                if stable_steps >= SETTLE_STABLE_STEPS:
                    settled = True
                    settle_steps = settle_step
                    break
            else:
                stable_steps = 0
        outcome = "placed" if settled else "collapse"
        if outcome == "collapse" and continue_after_collapse:
            sequence, simulated_steps = self._settle_collapse_tail(
                frames, sequence, simulated_steps, frame_callback
            )
        if outcome == "placed" and not self._available_positions_for_layer(layer):
            self.placement_layer = None
            self.placement_anchor = None
            self.placement_surface_z = None
        self._zero_velocities()
        final = self.frame(
            sequence=sequence + 1,
            sim_time=simulated_steps * TIMESTEP,
            phase=outcome,
        )
        frames.append(final)
        self._emit_frame(final, frame_callback)
        self.last_frames = tuple(frames)
        return PlaceResult(
            outcome=outcome,
            frames=self.last_frames,
            target_id=placed.spec.internal_id,
            settle_steps=settle_steps,
        )

    def _validate_place(self, request: PlaceRequest) -> BlockBody:
        if self.held_block is None:
            raise PlaceValidationError("PlaceBack requires an extracted block")
        if request.position not in ("Left", "Middle", "Right"):
            raise PlaceValidationError("position must be Left, Middle, or Right")
        if request.position not in self.available_placement_positions:
            raise PlaceValidationError("position is already occupied")
        return self.held_block

    def _next_placement_layer(self) -> int:
        if self.placement_layer is None:
            self.placement_layer = self.top_layer + 1
        return self.placement_layer

    def _placement_row_anchor(self) -> tuple[float, float]:
        if self.placement_anchor is None:
            anchor_layer = max(self.top_layer - 1, 1)
            positions = [
                bullet.getBasePositionAndOrientation(block.body_id, physicsClientId=self.client_id)[0]
                for block in self.blocks
                if block.body_id not in self.retired_body_ids and block.spec.layer == anchor_layer
            ]
            self.placement_anchor = (
                sum(position[0] for position in positions) / len(positions),
                sum(position[1] for position in positions) / len(positions),
            )
        return self.placement_anchor

    def _placement_row_surface(self) -> float:
        if self.placement_surface_z is None:
            self.placement_surface_z = max(
                bullet.getBasePositionAndOrientation(block.body_id, physicsClientId=self.client_id)[0][2]
                + BLOCK_HEIGHT / 2
                for block in self.blocks
                if block.body_id not in self.retired_body_ids
            )
        return self.placement_surface_z

    @staticmethod
    def _orientation_for_layer(layer: int) -> Orientation:
        return Orientation.NORTH_SOUTH if layer % 2 else Orientation.EAST_WEST

    @staticmethod
    def _block_size(spec: BlockSpec) -> tuple[float, float, float]:
        length, width, height = spec.dimensions
        return (
            (width, length, height)
            if spec.orientation == Orientation.NORTH_SOUTH
            else (length, width, height)
        )

    @staticmethod
    def _position_for_slot(orientation: Orientation, slot: str) -> str:
        names = NORTH_SOUTH_SLOTS if orientation == Orientation.NORTH_SOUTH else EAST_WEST_SLOTS
        return ("Left", "Middle", "Right")[[value.name for value in names].index(slot)]

    def _world_contact_point(self, target: BlockBody, request: PushRequest) -> tuple[float, ...]:
        lateral = {"left": -1.0, "center": 0.0, "right": 1.0}[request.contact] * (target.spec.dimensions[1] / 3)
        vertical = 0.0
        if request.face == "North":
            local = (lateral, target.spec.dimensions[0] / 2, vertical)
        elif request.face == "South":
            local = (-lateral, -target.spec.dimensions[0] / 2, vertical)
        elif request.face == "East":
            local = (target.spec.dimensions[0] / 2, lateral, vertical)
        else:
            local = (-target.spec.dimensions[0] / 2, -lateral, vertical)
        position, rotation = bullet.getBasePositionAndOrientation(target.body_id, physicsClientId=self.client_id)
        world, _ = bullet.multiplyTransforms(position, rotation, local, (0.0, 0.0, 0.0, 1.0))
        return tuple(world)

    def _get_load(self, body_id: int) -> float:
        contacts = bullet.getContactPoints(bodyA=body_id, physicsClientId=self.client_id)
        # Only count vertical contacts (weight from above/below), not side neighbors
        return max(sum(c[9] for c in contacts if abs(c[7][2]) > 0.7), 0.01)

    def _force_direction_aligned(self, target: BlockBody, face: str) -> tuple[float, float, float]:
        local_dir = {
            "North": (0.0, -1.0, 0.0),
            "South": (0.0, 1.0, 0.0),
            "East": (-1.0, 0.0, 0.0),
            "West": (1.0, 0.0, 0.0),
        }[face]
        _, rotation = bullet.getBasePositionAndOrientation(target.body_id, physicsClientId=self.client_id)
        world_dir, _ = bullet.multiplyTransforms((0, 0, 0), rotation, local_dir, (0, 0, 0, 1))
        return (world_dir[0], world_dir[1], 0.0)

    def _is_extracted(self, target: BlockBody) -> bool:
        block_ids = {b.body_id for b in self.blocks if b.body_id != target.body_id and b.body_id not in self.retired_body_ids}
        contacts = bullet.getContactPoints(bodyA=target.body_id, physicsClientId=self.client_id)
        return not any(c[2] in block_ids for c in contacts)

    def _snapshot_vertical_contacts(self, target_id: int) -> dict[int, set[int]]:
        snapshot: dict[int, set[int]] = {}
        for block in self.blocks:
            if block.body_id == target_id or block.body_id in self.retired_body_ids:
                continue
            contacts = bullet.getContactPoints(bodyA=block.body_id, physicsClientId=self.client_id)
            vertical = {c[2] for c in contacts if abs(c[7][2]) > 0.7 and c[2] != target_id}
            if vertical:
                snapshot[block.body_id] = vertical
        return snapshot

    def _check_collapse(self, snapshot: dict[int, set[int]], lost_steps: dict[int, int], target_id: int = -1) -> bool:
        ground_ids = {self.base_body_id, self.floor_body_id}
        for block in self.blocks:
            if block.body_id in self.retired_body_ids or block.spec.layer == 1 or block.body_id == target_id:
                continue
            contacts = bullet.getContactPoints(bodyA=block.body_id, physicsClientId=self.client_id)
            if any(c[2] in ground_ids for c in contacts):
                logger.debug("COLLAPSE: %s (layer %d) touched ground/base", block.spec.internal_id, block.spec.layer)
                return True
        for body_id, required in snapshot.items():
            contacts = bullet.getContactPoints(bodyA=body_id, physicsClientId=self.client_id)
            current = {c[2] for c in contacts if abs(c[7][2]) > 0.7}
            lost = required - current
            if lost:
                lost_steps[body_id] = lost_steps.get(body_id, 0) + 1
                if lost_steps[body_id] >= COLLAPSE_CONSECUTIVE_STEPS:
                    spec = next((b.spec for b in self.blocks if b.body_id == body_id), None)
                    lost_names = []
                    for lid in lost:
                        lspec = next((b.spec for b in self.blocks if b.body_id == lid), None)
                        lost_names.append(lspec.internal_id if lspec else str(lid))
                    logger.debug(
                        "COLLAPSE: %s (layer %d) lost vertical contact with [%s] for %d steps. had=%s now=%s",
                        spec.internal_id if spec else body_id,
                        spec.layer if spec else -1,
                        ", ".join(lost_names),
                        lost_steps[body_id],
                        required,
                        current,
                    )
                    return True
            else:
                lost_steps[body_id] = 0
        return False

    def _settle_with_frames(
        self,
        frames: list[dict[str, Any]],
        sequence: int,
        simulated_steps: int,
        target: BlockBody,
        extracted: bool,
        contact_snapshot: dict[int, set[int]],
        collapse_lost_steps: dict[int, int],
        frame_callback: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[bool, bool, int, int, int]:
        stable_steps = 0
        timeout_steps = round(SETTLE_TIMEOUT_SECONDS / TIMESTEP)
        for settle_step in range(1, timeout_steps + 1):
            bullet.stepSimulation(physicsClientId=self.client_id)
            simulated_steps += 1
            if settle_step % FRAME_SAMPLE_STEPS == 0:
                sequence += 1
                frames.append(self.frame(sequence=sequence, sim_time=simulated_steps * TIMESTEP, phase="settle"))
                self._emit_frame(frames[-1], frame_callback)
            extracted = extracted or self._is_extracted(target)
            if self._check_collapse(contact_snapshot, collapse_lost_steps, target.body_id):
                return False, extracted, sequence, simulated_steps, settle_step
            if self._all_blocks_below_velocity_thresholds(ignored_body_id=target.body_id if extracted else None):
                stable_steps += 1
                if stable_steps >= SETTLE_STABLE_STEPS:
                    return True, extracted, sequence, simulated_steps, settle_step
            else:
                stable_steps = 0
        return False, extracted, sequence, simulated_steps, timeout_steps

    def _settle_collapse_tail(
        self,
        frames: list[dict[str, Any]],
        sequence: int,
        simulated_steps: int,
        frame_callback: Callable[[dict[str, Any]], None] | None,
    ) -> tuple[int, int]:
        stable_steps = 0
        timeout_steps = round(VIEWER_COLLAPSE_TAIL_TIMEOUT_SECONDS / TIMESTEP)
        for tail_step in range(1, timeout_steps + 1):
            bullet.stepSimulation(physicsClientId=self.client_id)
            simulated_steps += 1
            if tail_step % FRAME_SAMPLE_STEPS == 0:
                sequence += 1
                frames.append(
                    self.frame(
                        sequence=sequence,
                        sim_time=simulated_steps * TIMESTEP,
                        phase="collapse",
                    )
                )
                self._emit_frame(frames[-1], frame_callback)
            if self._all_blocks_below_velocity_thresholds(
                linear_threshold=VIEWER_COLLAPSE_LINEAR_VELOCITY_THRESHOLD,
                angular_threshold=VIEWER_COLLAPSE_ANGULAR_VELOCITY_THRESHOLD,
            ):
                stable_steps += 1
                if stable_steps >= VIEWER_COLLAPSE_STABLE_STEPS:
                    break
            else:
                stable_steps = 0
        return sequence, simulated_steps

    @staticmethod
    def _emit_frame(
        frame: dict[str, Any], frame_callback: Callable[[dict[str, Any]], None] | None
    ) -> None:
        if frame_callback is not None:
            frame_callback(frame)

    def _configure(self) -> None:
        bullet.setGravity(*PHYSICS.gravity, physicsClientId=self.client_id)
        bullet.setTimeStep(TIMESTEP, physicsClientId=self.client_id)
        bullet.setPhysicsEngineParameter(
            fixedTimeStep=TIMESTEP,
            numSolverIterations=SOLVER_ITERATIONS,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client_id,
        )

    def _clear_world(self) -> None:
        bullet.resetSimulation(physicsClientId=self.client_id)
        self._configure()
        self.blocks = ()
        self.base_body_id = None
        self.floor_body_id = None
        self.settle_steps = None
        self.last_frames = ()
        self.retired_body_ids = set()
        self.held_block = None
        self.placement_layer = None
        self.placement_anchor = None
        self.placement_surface_z = None

    def _build_world(self, specs: tuple[BlockSpec, ...]) -> None:
        floor_half = tuple(value / 2 for value in FLOOR_SIZE)
        floor_shape = bullet.createCollisionShape(
            bullet.GEOM_BOX, halfExtents=floor_half, physicsClientId=self.client_id
        )
        floor_visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=floor_half,
            rgbaColor=(0.59, 0.39, 0.26, 1.0),
            physicsClientId=self.client_id,
        )
        self.floor_body_id = bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_shape,
            baseVisualShapeIndex=floor_visual,
            basePosition=(0.0, 0.0, FLOOR_CENTER_Z),
            physicsClientId=self.client_id,
        )
        bullet.changeDynamics(
            self.floor_body_id, -1,
            restitution=PHYSICS.floor_restitution,
            physicsClientId=self.client_id,
        )
        base_half = tuple(value / 2 for value in BASE_SIZE)
        base_collision = bullet.createCollisionShape(
            bullet.GEOM_BOX, halfExtents=base_half, physicsClientId=self.client_id
        )
        base_visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=base_half,
            rgbaColor=(0.35, 0.22, 0.14, 1.0),
            physicsClientId=self.client_id,
        )
        self.base_body_id = bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=(0.0, 0.0, BASE_CENTER_Z),
            physicsClientId=self.client_id,
        )
        bullet.changeDynamics(
            self.base_body_id,
            -1,
            lateralFriction=PHYSICS.lateral_friction,
            restitution=PHYSICS.floor_restitution,
            physicsClientId=self.client_id,
        )
        bodies = []
        for spec in specs:
            bodies.append(self._create_block_body(spec))
        self.blocks = tuple(bodies)

    def _create_block_body(self, spec: BlockSpec) -> BlockBody:
        dimensions = self._block_size(spec)
        half_extents = tuple(value / 2 for value in dimensions)
        collision = bullet.createCollisionShape(
            bullet.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id
        )
        visual = bullet.createVisualShape(
            bullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=spec.rgba, physicsClientId=self.client_id
        )
        body_id = bullet.createMultiBody(
            baseMass=BLOCK_MASS,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=spec.position,
            baseOrientation=bullet.getQuaternionFromEuler((0.0, 0.0, math.radians(spec.yaw_degrees))),
            physicsClientId=self.client_id,
        )
        bullet.changeDynamics(
            body_id,
            -1,
            lateralFriction=PHYSICS.lateral_friction,
            rollingFriction=PHYSICS.rolling_friction,
            spinningFriction=PHYSICS.spinning_friction,
            linearDamping=PHYSICS.linear_damping,
            angularDamping=PHYSICS.angular_damping,
            restitution=PHYSICS.restitution,
            physicsClientId=self.client_id,
        )
        return BlockBody(spec=spec, body_id=body_id)

    def _settle(self) -> bool:
        stable_steps = 0
        timeout_steps = round(SETTLE_TIMEOUT_SECONDS / TIMESTEP)
        for step in range(1, timeout_steps + 1):
            bullet.stepSimulation(physicsClientId=self.client_id)
            if self._all_blocks_below_velocity_thresholds():
                stable_steps += 1
                if stable_steps >= SETTLE_STABLE_STEPS:
                    self.settle_steps = step
                    return True
            else:
                stable_steps = 0
        self.settle_steps = timeout_steps
        return False

    def _zero_velocities(self) -> None:
        for block in self.blocks:
            if block.body_id in self.retired_body_ids:
                continue
            bullet.resetBaseVelocity(
                block.body_id,
                linearVelocity=(0.0, 0.0, 0.0),
                angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=self.client_id,
            )

    def _all_blocks_below_velocity_thresholds(
        self,
        *,
        ignored_body_id: int | None = None,
        linear_threshold: float = LINEAR_VELOCITY_THRESHOLD,
        angular_threshold: float = ANGULAR_VELOCITY_THRESHOLD,
    ) -> bool:
        for block in self.blocks:
            if block.body_id == ignored_body_id or block.body_id in self.retired_body_ids:
                continue
            linear, angular = bullet.getBaseVelocity(block.body_id, physicsClientId=self.client_id)
            if max(abs(value) for value in linear) >= linear_threshold:
                return False
            if max(abs(value) for value in angular) >= angular_threshold:
                return False
        return True
