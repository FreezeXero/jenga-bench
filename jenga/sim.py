"""Authoritative deterministic PyBullet simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pybullet as bullet

from jenga.tower import (
    ANGULAR_DAMPING,
    BASE_CENTER_Z,
    BASE_SIZE,
    BLOCK_HEIGHT,
    BLOCK_LENGTH,
    BLOCK_MASS,
    FLOOR_CENTER_Z,
    FLOOR_SIZE,
    LATERAL_FRICTION,
    LINEAR_DAMPING,
    ROLLING_FRICTION,
    SPINNING_FRICTION,
    BlockSpec,
    Orientation,
    build_prebuilt_tower,
)

TIMESTEP = 1.0 / 240.0
SOLVER_ITERATIONS = 100
SETTLE_TIMEOUT_SECONDS = 3.0
SETTLE_STABLE_STEPS = 30
LINEAR_VELOCITY_THRESHOLD = 2e-3
ANGULAR_VELOCITY_THRESHOLD = 2e-2
RAMP_DURATION_SECONDS = 0.4
RAMP_STEPS = round(RAMP_DURATION_SECONDS / TIMESTEP)
FRAME_SAMPLE_STEPS = 8
MAX_TILT_DEGREES = 20.0
BASE_HALF_WIDTH = BASE_SIZE[0] / 2

CONTACTS = (
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)
INTENSITIES = {"Gentle": 1.5, "Firm": 3, "Hard": 5}
VALID_FACES = {
    Orientation.NORTH_SOUTH: ("North", "South"),
    Orientation.EAST_WEST: ("East", "West"),
}


class TowerStabilityError(RuntimeError):
    """Raised when the deterministic tower cannot settle."""


class PushValidationError(ValueError):
    """Raised when a push request is invalid."""


@dataclass(frozen=True)
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


class JengaSimulation:
    """Owns one PyBullet DIRECT client and its dynamic tower."""

    def __init__(self) -> None:
        self.client_id = bullet.connect(bullet.DIRECT)
        self.blocks: tuple[BlockBody, ...] = ()
        self.base_body_id: int | None = None
        self.floor_body_id: int | None = None
        self.settle_steps: int | None = None
        self.last_frames: tuple[dict[str, Any], ...] = ()
        self._configure()

    @property
    def is_connected(self) -> bool:
        return bool(bullet.isConnected(self.client_id))

    def close(self) -> None:
        if self.is_connected:
            bullet.disconnect(self.client_id)

    def reset(self, seed: int | None) -> None:
        del seed
        self._clear_world()
        self._build_world(build_prebuilt_tower())
        if not self._settle():
            raise TowerStabilityError("prebuilt tower failed to settle")
        self._zero_velocities()
        self.last_frames = (self.frame(sequence=0, sim_time=0.0, phase="initial"),)

    def transforms(self) -> tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]:
        values = []
        for block in self.blocks:
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
                {"id": internal_id, "position": position, "rotation": rotation}
                for internal_id, position, rotation in self.transforms()
            ],
        }

    def push(self, request: PushRequest) -> PushResult:
        target = self._validate_push(request)
        initial_position, _ = bullet.getBasePositionAndOrientation(
            target.body_id, physicsClientId=self.client_id
        )
        frames = [self.frame(sequence=0, sim_time=0.0, phase="initial")]
        sequence = 0
        simulated_steps = 0

        direction = self._force_direction(request.face)
        contact = self._world_contact_point(target, request)
        peak_force = INTENSITIES[request.intensity]
        for ramp_step in range(1, RAMP_STEPS + 1):
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
            if self._has_obvious_collapse(target.body_id):
                return self._finish_push("collapse", target, frames, sequence, simulated_steps, 0)

        settled, sequence, simulated_steps, settle_steps = self._settle_with_frames(
            frames, sequence, simulated_steps, target.body_id
        )
        if not settled or self._has_obvious_collapse(target.body_id):
            outcome = "collapse"
        elif self._is_extracted(target, initial_position, direction):
            outcome = "extracted"
        else:
            outcome = "settled"
        self._zero_velocities()
        return self._finish_push(outcome, target, frames, sequence, simulated_steps, settle_steps)

    def _finish_push(
        self,
        outcome: str,
        target: BlockBody,
        frames: list[dict[str, Any]],
        sequence: int,
        simulated_steps: int,
        settle_steps: int,
    ) -> PushResult:
        self._zero_velocities()
        final_sequence = sequence + 1
        final = self.frame(sequence=final_sequence, sim_time=simulated_steps * TIMESTEP, phase=outcome)
        if frames[-1]["blocks"] == final["blocks"]:
            final["sequence"] = frames[-1]["sequence"]
            frames[-1] = final
        else:
            frames.append(final)
        self.last_frames = tuple(frames)
        return PushResult(
            outcome=outcome,
            frames=self.last_frames,
            target_id=target.spec.internal_id,
            ramp_steps=RAMP_STEPS,
            settle_steps=settle_steps,
        )

    def _validate_push(self, request: PushRequest) -> BlockBody:
        if isinstance(request.layer, bool) or not isinstance(request.layer, int) or not 1 <= request.layer <= 18:
            raise PushValidationError("layer must be an integer between 1 and 18")
        target = next(
            (
                block
                for block in self.blocks
                if block.spec.layer == request.layer and block.spec.color_name == request.color
            ),
            None,
        )
        if target is None:
            raise PushValidationError("color does not identify a block in the requested layer")
        if request.face not in VALID_FACES[target.spec.orientation]:
            valid = ", ".join(VALID_FACES[target.spec.orientation])
            raise PushValidationError(f"face must be one of: {valid}")
        if request.contact not in CONTACTS:
            raise PushValidationError("contact must be one of the nine 3x3 grid positions")
        if request.intensity not in INTENSITIES:
            raise PushValidationError("intensity must be Gentle, Firm, or Hard")
        return target

    def _world_contact_point(self, target: BlockBody, request: PushRequest) -> tuple[float, ...]:
        row, column = request.contact.split("-") if "-" in request.contact else ("center", "center")
        lateral = {"left": -1.0, "center": 0.0, "right": 1.0}[column] * (target.spec.dimensions[1] / 3)
        vertical = {"bottom": -1.0, "center": 0.0, "top": 1.0}[row] * (BLOCK_HEIGHT / 3)
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

    @staticmethod
    def _force_direction(face: str) -> tuple[float, float, float]:
        return {
            "North": (0.0, -1.0, 0.0),
            "South": (0.0, 1.0, 0.0),
            "East": (-1.0, 0.0, 0.0),
            "West": (1.0, 0.0, 0.0),
        }[face]

    def _is_extracted(
        self, target: BlockBody, initial_position: tuple[float, ...], direction: tuple[float, ...]
    ) -> bool:
        position, _ = bullet.getBasePositionAndOrientation(target.body_id, physicsClientId=self.client_id)
        movement = sum((position[index] - initial_position[index]) * direction[index] for index in range(3))
        return movement >= BLOCK_LENGTH

    def _has_obvious_collapse(self, target_id: int) -> bool:
        for block in self.blocks:
            if block.body_id == target_id:
                continue
            position, rotation = bullet.getBasePositionAndOrientation(
                block.body_id, physicsClientId=self.client_id
            )
            roll, pitch, _ = bullet.getEulerFromQuaternion(rotation)
            if position[2] < 0.0:
                return True
            if abs(position[0]) > BASE_HALF_WIDTH or abs(position[1]) > BASE_HALF_WIDTH:
                return True
            if math.degrees(max(abs(roll), abs(pitch))) > MAX_TILT_DEGREES:
                return True
        return False

    def _settle_with_frames(
        self,
        frames: list[dict[str, Any]],
        sequence: int,
        simulated_steps: int,
        target_id: int,
    ) -> tuple[bool, int, int, int]:
        stable_steps = 0
        timeout_steps = round(SETTLE_TIMEOUT_SECONDS / TIMESTEP)
        for settle_step in range(1, timeout_steps + 1):
            bullet.stepSimulation(physicsClientId=self.client_id)
            simulated_steps += 1
            if settle_step % FRAME_SAMPLE_STEPS == 0:
                sequence += 1
                frames.append(self.frame(sequence=sequence, sim_time=simulated_steps * TIMESTEP, phase="settle"))
            if self._has_obvious_collapse(target_id):
                return False, sequence, simulated_steps, settle_step
            if self._all_blocks_below_velocity_thresholds():
                stable_steps += 1
                if stable_steps >= SETTLE_STABLE_STEPS:
                    return True, sequence, simulated_steps, settle_step
            else:
                stable_steps = 0
        return False, sequence, simulated_steps, timeout_steps

    def _configure(self) -> None:
        bullet.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
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

    def _build_world(self, specs: tuple[BlockSpec, ...]) -> None:
        floor_half = tuple(value / 2 for value in FLOOR_SIZE)
        floor_shape = bullet.createCollisionShape(
            bullet.GEOM_BOX, halfExtents=floor_half, physicsClientId=self.client_id
        )
        floor_visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=floor_half,
            rgbaColor=(0.96, 0.96, 0.96, 1.0),
            physicsClientId=self.client_id,
        )
        self.floor_body_id = bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_shape,
            baseVisualShapeIndex=floor_visual,
            basePosition=(0.0, 0.0, FLOOR_CENTER_Z),
            physicsClientId=self.client_id,
        )
        base_half = tuple(value / 2 for value in BASE_SIZE)
        base_collision = bullet.createCollisionShape(
            bullet.GEOM_BOX, halfExtents=base_half, physicsClientId=self.client_id
        )
        base_visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=base_half,
            rgbaColor=(0.03, 0.03, 0.035, 1.0),
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
            self.base_body_id, -1, lateralFriction=LATERAL_FRICTION, physicsClientId=self.client_id
        )
        bodies = []
        for spec in specs:
            length, width, height = spec.dimensions
            dimensions = (width, length, height) if spec.orientation == Orientation.NORTH_SOUTH else (length, width, height)
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
                physicsClientId=self.client_id,
            )
            bullet.changeDynamics(
                body_id,
                -1,
                lateralFriction=LATERAL_FRICTION,
                rollingFriction=ROLLING_FRICTION,
                spinningFriction=SPINNING_FRICTION,
                linearDamping=LINEAR_DAMPING,
                angularDamping=ANGULAR_DAMPING,
                restitution=0.0,
                physicsClientId=self.client_id,
            )
            bodies.append(BlockBody(spec=spec, body_id=body_id))
        self.blocks = tuple(bodies)

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
            bullet.resetBaseVelocity(
                block.body_id,
                linearVelocity=(0.0, 0.0, 0.0),
                angularVelocity=(0.0, 0.0, 0.0),
                physicsClientId=self.client_id,
            )

    def _all_blocks_below_velocity_thresholds(self) -> bool:
        for block in self.blocks:
            linear, angular = bullet.getBaseVelocity(block.body_id, physicsClientId=self.client_id)
            if max(abs(value) for value in linear) >= LINEAR_VELOCITY_THRESHOLD:
                return False
            if max(abs(value) for value in angular) >= ANGULAR_VELOCITY_THRESHOLD:
                return False
        return True
