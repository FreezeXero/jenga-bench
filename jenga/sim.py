"""Authoritative headless PyBullet simulation for the static tower slice."""

from __future__ import annotations

from dataclasses import dataclass
import pybullet as bullet

from jenga.tower import (
    BASE_CENTER_Z,
    BASE_SIZE,
    BLOCK_MASS,
    FLOOR_CENTER_Z,
    FLOOR_SIZE,
    LATERAL_FRICTION,
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
LINEAR_VELOCITY_THRESHOLD = 1e-3
ANGULAR_VELOCITY_THRESHOLD = 1e-3
@dataclass(frozen=True)
class BlockBody:
    spec: BlockSpec
    body_id: int


class JengaSimulation:
    """Owns one PyBullet DIRECT client and its accepted static tower."""

    def __init__(self) -> None:
        self.client_id = bullet.connect(bullet.DIRECT)
        self.blocks: tuple[BlockBody, ...] = ()
        self.base_body_id: int | None = None
        self.floor_body_id: int | None = None
        self.settle_steps: int | None = None
        self._configure()

    @property
    def is_connected(self) -> bool:
        return bool(bullet.isConnected(self.client_id))

    def close(self) -> None:
        if self.is_connected:
            bullet.disconnect(self.client_id)

    def reset(self, seed: int | None) -> None:
        del seed
        specs = build_prebuilt_tower()
        self._clear_world()
        self._build_world(specs)
        # Milestone 2 loads an exact static snapshot. Physics stepping begins
        # when interaction actions are introduced in a later milestone.
        self.settle_steps = 0

    def transforms(self) -> tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]:
        values = []
        for block in self.blocks:
            position, rotation = bullet.getBasePositionAndOrientation(
                block.body_id, physicsClientId=self.client_id
            )
            values.append((block.spec.internal_id, tuple(position), tuple(rotation)))
        return tuple(values)

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

        bodies = []
        for spec in specs:
            long_side, short_side, height = spec.dimensions
            if spec.orientation == Orientation.NORTH_SOUTH:
                dimensions = (short_side, long_side, height)
            else:
                dimensions = (long_side, short_side, height)
            half_extents = tuple(value / 2 for value in dimensions)
            collision = bullet.createCollisionShape(
                bullet.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id
            )
            visual = bullet.createVisualShape(
                bullet.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=spec.rgba,
                physicsClientId=self.client_id,
            )
            orientation = bullet.getQuaternionFromEuler((0.0, 0.0, 0.0))
            body_id = bullet.createMultiBody(
                baseMass=BLOCK_MASS,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=spec.position,
                baseOrientation=orientation,
                physicsClientId=self.client_id,
            )
            bullet.changeDynamics(
                body_id,
                -1,
                lateralFriction=LATERAL_FRICTION,
                rollingFriction=ROLLING_FRICTION,
                spinningFriction=SPINNING_FRICTION,
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

    def _all_blocks_below_velocity_thresholds(self) -> bool:
        for block in self.blocks:
            linear, angular = bullet.getBaseVelocity(block.body_id, physicsClientId=self.client_id)
            if max(abs(value) for value in linear) >= LINEAR_VELOCITY_THRESHOLD:
                return False
            if max(abs(value) for value in angular) >= ANGULAR_VELOCITY_THRESHOLD:
                return False
        return True
