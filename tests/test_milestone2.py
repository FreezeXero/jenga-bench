from __future__ import annotations

import importlib.util
import struct
import unittest

from jenga.tower import (
    BASE_CENTER_Z,
    BASE_SIZE,
    BLOCK_HEIGHT,
    BLOCK_LENGTH,
    BLOCK_WIDTH,
    BLOCKS_PER_LAYER,
    EAST_WEST_SLOTS,
    FLOOR_CENTER_Z,
    FLOOR_SIZE,
    FLOOR_Z,
    LAYER_COUNT,
    NORTH_SOUTH_SLOTS,
    Orientation,
    build_prebuilt_tower,
)

PHYSICS_AVAILABLE = bool(importlib.util.find_spec("pybullet"))

if PHYSICS_AVAILABLE:
    from env import JengaBenchEnv
    from jenga.render import IMAGE_HEIGHT, IMAGE_WIDTH
    from jenga.sim import JengaSimulation


class TowerSpecificationTests(unittest.TestCase):
    def test_prebuilt_tower_contains_expected_layers_slots_and_colors(self) -> None:
        specs = build_prebuilt_tower()

        self.assertEqual(len(specs), LAYER_COUNT * BLOCKS_PER_LAYER)
        self.assertEqual(len({spec.internal_id for spec in specs}), len(specs))
        for layer in range(1, LAYER_COUNT + 1):
            blocks = [spec for spec in specs if spec.layer == layer]
            expected_orientation = (
                Orientation.NORTH_SOUTH if layer % 2 == 1 else Orientation.EAST_WEST
            )
            expected_slots = (
                NORTH_SOUTH_SLOTS
                if expected_orientation == Orientation.NORTH_SOUTH
                else EAST_WEST_SLOTS
            )
            self.assertEqual([block.orientation for block in blocks], [expected_orientation] * 3)
            self.assertEqual([block.slot for block in blocks], [slot.name for slot in expected_slots])
            self.assertEqual([block.rgb for block in blocks], [slot.rgb for slot in expected_slots])

    def test_prebuilt_tower_uses_exact_geometry(self) -> None:
        specs = build_prebuilt_tower()

        for spec in specs:
            self.assertEqual(spec.dimensions, (BLOCK_LENGTH, BLOCK_WIDTH, BLOCK_HEIGHT))
            self.assertEqual(spec.yaw_degrees, 0.0)
            if spec.orientation == Orientation.NORTH_SOUTH:
                expected_offset = next(slot.offset for slot in NORTH_SOUTH_SLOTS if slot.name == spec.slot)
                self.assertEqual(spec.position[0], expected_offset)
                self.assertEqual(spec.position[1], 0.0)
            else:
                expected_offset = next(slot.offset for slot in EAST_WEST_SLOTS if slot.name == spec.slot)
                self.assertEqual(spec.position[0], 0.0)
                self.assertEqual(spec.position[1], expected_offset)

    def test_base_and_floor_coordinates(self) -> None:
        self.assertEqual(BASE_CENTER_Z + BASE_SIZE[2] / 2, 0.0)
        self.assertLess(FLOOR_Z, BASE_CENTER_Z)
        self.assertEqual(FLOOR_CENTER_Z + FLOOR_SIZE[2] / 2, FLOOR_Z)


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class SimulationTests(unittest.TestCase):
    def test_prebuilt_tower_settles_dynamically(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            self.assertGreater(sim.settle_steps, 0)
            self.assertLessEqual(sim.settle_steps, 720)
        finally:
            sim.close()

    def test_different_seeds_have_identical_transforms_and_png(self) -> None:
        first = JengaBenchEnv()
        second = JengaBenchEnv()
        try:
            first_image = first.reset(seed=1)["data"]
            second_image = second.reset(seed=999)["data"]
            self.assertEqual(first._simulation.transforms(), second._simulation.transforms())
            self.assertEqual(first_image, second_image)
        finally:
            first.close()
            second.close()

    def test_viewpoint_changes_image_without_moving_tower(self) -> None:
        env = JengaBenchEnv()
        try:
            initial = env.reset(seed=4)["data"]
            transforms = env._simulation.transforms()
            result = env.step(
                {"type": "ChangeViewpoint", "azimuth": 90, "pitch": 5, "distance_cm": 60}
            )
            self.assertNotEqual(initial, result.observation)
            self.assertEqual(transforms, env._simulation.transforms())
            width, height = struct.unpack(">II", result.observation[16:24])
            self.assertEqual((width, height), (IMAGE_WIDTH, IMAGE_HEIGHT))
        finally:
            env.close()

    def test_close_disconnects_client(self) -> None:
        env = JengaBenchEnv()
        env.reset(seed=2)
        sim = env._simulation
        env.close()
        self.assertFalse(sim.is_connected)


if __name__ == "__main__":
    unittest.main()
