"""Central tuning presets for tower generation, physics, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TowerGeometrySettings:
    block_length: float = 0.075
    block_width: float = 0.025
    block_height: float = 0.015
    block_mass: float = 0.120
    block_clearance: float = 0.0002
    layer_count: int = 18
    blocks_per_layer: int = 3
    base_size: tuple[float, float, float] = (0.25, 0.25, 0.045)
    floor_size: tuple[float, float, float] = (4.0, 4.0, 0.01)


@dataclass(frozen=True)
class TowerRandomnessSettings:
    block_longitudinal_offset: float = 0.0050
    layer_shift_step: float = 0.0015
    layer_yaw_degrees: float = 4
    extra_layer_gap: float = 0.0010


@dataclass(frozen=True)
class PhysicsSettings:
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 1.0 / 240.0
    solver_iterations: int = 100
    lateral_friction: float = 0.50
    rolling_friction: float = 0.0
    spinning_friction: float = 0.0
    linear_damping: float = 0.04
    angular_damping: float = 0.04
    settle_timeout_seconds: float = 10.0
    viewer_collapse_tail_timeout_seconds: float = 5.0
    viewer_collapse_linear_velocity_threshold: float = 0.02
    viewer_collapse_angular_velocity_threshold: float = 0.15
    viewer_collapse_stable_steps: int = 8
    settle_stable_steps: int = 30
    linear_velocity_threshold: float = 5e-3
    angular_velocity_threshold: float = 5e-2
    ramp_duration_seconds: float = 0.2
    placement_drop_height: float = 0.005
    max_placement_rotation_degrees: float = 5.0
    frame_sample_steps: int = 8
    max_tilt_degrees: float = 20.0
    push_force_multiplier: float = 1.5
    intensities: tuple[tuple[str, float], ...] = (
        ("Gentle", 0.05),
        ("Firm", 0.15),
        ("Hard", 0.40),
    )


@dataclass(frozen=True)
class RenderSettings:
    image_width: int = 512
    image_height: int = 512
    tower_midpoint: tuple[float, float, float] = (0.0, 0.0, 0.135)
    field_of_view_degrees: float = 52.0
    near_plane: float = 0.02
    far_plane: float = 3.0
    light_direction: tuple[float, float, float] = (3.0, -4.0, 6.0)
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_ambient_coefficient: float = 0.7
    light_diffuse_coefficient: float = 0.6
    light_specular_coefficient: float = 0.05


@dataclass(frozen=True)
class JengaSettings:
    geometry: TowerGeometrySettings = field(default_factory=TowerGeometrySettings)
    randomness: TowerRandomnessSettings = field(default_factory=TowerRandomnessSettings)
    physics: PhysicsSettings = field(default_factory=PhysicsSettings)
    render: RenderSettings = field(default_factory=RenderSettings)


DEFAULT_SETTINGS = JengaSettings()
