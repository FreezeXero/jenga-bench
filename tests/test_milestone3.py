from __future__ import annotations

import importlib.util
import json
import unittest

PHYSICS_AVAILABLE = bool(importlib.util.find_spec("pybullet"))

if PHYSICS_AVAILABLE:
    import pybullet as bullet
    from fastapi.testclient import TestClient

    from env import JengaBenchEnv
    from jenga.sim import (
        FRAME_SAMPLE_STEPS,
        INTENSITIES,
        RAMP_STEPS,
        JengaSimulation,
        PlaceRequest,
        PlaceValidationError,
        PushRequest,
        PushValidationError,
    )
    from showcase.server import app, motion_lock, preview

    REQUEST = PushRequest(layer=8, color="Purple", face="East", contact="center", intensity="Gentle")
else:
    REQUEST = None


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class DynamicTowerTests(unittest.TestCase):
    def test_untouched_tower_settles_and_remains_upright(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            self.assertLessEqual(sim.settle_steps, 720)
            for _ in range(720):
                bullet.stepSimulation(physicsClientId=sim.client_id)
            self.assertFalse(sim._has_obvious_collapse(target_id=-1))
            self.assertEqual(len(sim.blocks), 54)
        finally:
            sim.close()

    def test_same_push_is_deterministic(self) -> None:
        first = JengaSimulation()
        second = JengaSimulation()
        try:
            first.reset(seed=0)
            second.reset(seed=0)
            first_result = first.push(REQUEST)
            second_result = second.push(REQUEST)
            self.assertEqual(first_result, second_result)
            self.assertEqual(first.transforms(), second.transforms())
        finally:
            first.close()
            second.close()

    def test_push_frames_are_sampled_and_include_quaternions(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            result = sim.push(REQUEST)
            self.assertEqual(result.ramp_steps, RAMP_STEPS)
            self.assertEqual(RAMP_STEPS % FRAME_SAMPLE_STEPS, 0)
            self.assertEqual([frame["sequence"] for frame in result.frames], list(range(len(result.frames))))
            self.assertEqual(result.frames[-1]["phase"], result.outcome)
            self.assertTrue(
                all(len(block["rotation"]) == 4 for frame in result.frames for block in frame["blocks"])
            )
        finally:
            sim.close()

    def test_push_validation(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            invalid = [
                PushRequest(0, "Purple", "East", "center", "Gentle"),
                PushRequest(8, "Purple", "North", "center", "Gentle"),
                PushRequest(8, "Purple", "East", "outside", "Gentle"),
                PushRequest(8, "Purple", "East", "center", "Extreme"),
            ]
            for request in invalid:
                with self.subTest(request=request), self.assertRaises(PushValidationError):
                    sim.push(request)
        finally:
            sim.close()

    def test_top_layer_cannot_be_pushed(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            self.assertEqual(sim.max_push_layer, 17)
            sim._validate_push(PushRequest(17, "Lime", "North", "center", "Gentle"))
            with self.assertRaises(PushValidationError):
                sim.push(PushRequest(18, "Purple", "East", "center", "Gentle"))
        finally:
            sim.close()

    def test_force_levels_are_locked(self) -> None:
        self.assertEqual(INTENSITIES, {"Gentle": 1.5, "Firm": 3, "Hard": 5})

    def test_successful_extraction_requires_remaining_tower_to_settle(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            result = sim.push(PushRequest(10, "Purple", "East", "center", "Hard"))
            self.assertEqual(result.outcome, "extracted")
            self.assertEqual(result.frames[-1]["phase"], "extracted")
            self.assertGreater(result.settle_steps, 0)
            self.assertIn("settle", [frame["phase"] for frame in result.frames])
        finally:
            sim.close()

    def test_extracted_block_requires_place_back_before_next_push(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            extracted = sim.push(PushRequest(10, "Purple", "East", "center", "Hard"))
            self.assertEqual(extracted.outcome, "extracted")
            self.assertEqual(len(sim.retired_body_ids), 1)
            with self.assertRaises(PushValidationError):
                sim.push(PushRequest(8, "Purple", "East", "center", "Gentle"))
        finally:
            sim.close()

    def test_place_back_recolors_block_and_clears_held_state(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            extracted = sim.push(PushRequest(10, "Purple", "East", "center", "Hard"))
            self.assertEqual(extracted.outcome, "extracted")
            internal_id = extracted.target_id
            result = sim.place_back(PlaceRequest("Middle", 5))
            self.assertEqual(result.outcome, "placed")
            placed = next(block for block in sim.blocks if block.spec.internal_id == internal_id)
            self.assertEqual(placed.spec.layer, 19)
            self.assertEqual(placed.spec.color_name, "Lime")
            self.assertEqual(placed.spec.yaw_degrees, 5)
            self.assertIsNone(sim.held_block)
            self.assertIn("place-drop", [frame["phase"] for frame in result.frames])
            self.assertIn("color", result.frames[-1]["blocks"][0])
        finally:
            sim.close()

    def test_place_back_rejects_rotation_outside_alignment_lock(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            sim.push(PushRequest(10, "Purple", "East", "center", "Hard"))
            with self.assertRaises(PlaceValidationError):
                sim.place_back(PlaceRequest("Left", 5.01))
            self.assertIsNotNone(sim.held_block)
        finally:
            sim.close()

    def test_push_emits_frames_during_simulation(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            emitted = []
            result = sim.push(REQUEST, frame_callback=emitted.append)
            self.assertEqual(tuple(emitted), result.frames)
            self.assertEqual(emitted[0]["phase"], "initial")
            self.assertEqual(emitted[-1]["phase"], result.outcome)
        finally:
            sim.close()

    def test_viewer_collapse_tail_does_not_slow_scored_pushes(self) -> None:
        scored = JengaSimulation()
        viewer = JengaSimulation()
        try:
            scored.reset(seed=0)
            viewer.reset(seed=0)
            for sim in (scored, viewer):
                target = sim._validate_push(REQUEST)
                other = next(block for block in sim.blocks if block.body_id != target.body_id)
                position, rotation = bullet.getBasePositionAndOrientation(
                    other.body_id, physicsClientId=sim.client_id
                )
                bullet.resetBasePositionAndOrientation(
                    other.body_id,
                    (position[0], position[1], -0.01),
                    rotation,
                    physicsClientId=sim.client_id,
                )
            scored_result = scored.push(REQUEST)
            viewer_result = viewer.push(REQUEST, continue_after_collapse=True)
            self.assertEqual(scored_result.outcome, "collapse")
            self.assertEqual(viewer_result.outcome, "collapse")
            self.assertNotIn("collapse-settle", [frame["phase"] for frame in scored_result.frames])
            self.assertIn("collapse-settle", [frame["phase"] for frame in viewer_result.frames])
            self.assertGreater(len(viewer_result.frames), len(scored_result.frames))
        finally:
            scored.close()
            viewer.close()

    def test_obvious_collapse_and_extraction_geometry(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            target = sim._validate_push(REQUEST)
            initial, rotation = bullet.getBasePositionAndOrientation(
                target.body_id, physicsClientId=sim.client_id
            )
            bullet.resetBasePositionAndOrientation(
                target.body_id,
                (initial[0] - 0.08, initial[1], initial[2]),
                rotation,
                physicsClientId=sim.client_id,
            )
            self.assertTrue(sim._is_extracted(target, initial, (-1.0, 0.0, 0.0)))
            other = next(block for block in sim.blocks if block.body_id != target.body_id)
            position, rotation = bullet.getBasePositionAndOrientation(
                other.body_id, physicsClientId=sim.client_id
            )
            bullet.resetBasePositionAndOrientation(
                other.body_id,
                (position[0], position[1], -0.01),
                rotation,
                physicsClientId=sim.client_id,
            )
            self.assertTrue(sim._has_obvious_collapse(target.body_id))
        finally:
            sim.close()

    def test_transient_tilt_is_not_an_immediate_collapse(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            block = sim.blocks[0]
            position, _ = bullet.getBasePositionAndOrientation(
                block.body_id, physicsClientId=sim.client_id
            )
            tilted = bullet.getQuaternionFromEuler((0.0, 0.5, 0.0))
            bullet.resetBasePositionAndOrientation(
                block.body_id,
                position,
                tilted,
                physicsClientId=sim.client_id,
            )

            self.assertTrue(sim._has_obvious_collapse(target_id=-1))
            self.assertFalse(sim._has_obvious_collapse(target_id=-1, include_tilt=False))
        finally:
            sim.close()

    def test_collapse_takes_precedence_over_extraction(self) -> None:
        sim = JengaSimulation()
        try:
            sim.reset(seed=0)
            target = sim._validate_push(REQUEST)
            initial, rotation = bullet.getBasePositionAndOrientation(
                target.body_id, physicsClientId=sim.client_id
            )
            bullet.resetBasePositionAndOrientation(
                target.body_id,
                (initial[0] - 0.08, initial[1], initial[2]),
                rotation,
                physicsClientId=sim.client_id,
            )
            other = next(block for block in sim.blocks if block.body_id != target.body_id)
            position, rotation = bullet.getBasePositionAndOrientation(
                other.body_id, physicsClientId=sim.client_id
            )
            bullet.resetBasePositionAndOrientation(
                other.body_id,
                (position[0], position[1], -0.01),
                rotation,
                physicsClientId=sim.client_id,
            )

            result = sim.push(REQUEST)

            self.assertEqual(result.outcome, "collapse")
            self.assertIsNone(sim.held_block)
            self.assertEqual(sim.available_placement_positions, ())
        finally:
            sim.close()


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class ModelPushTests(unittest.TestCase):
    def test_model_push_returns_final_png_and_resets_view_counter(self) -> None:
        env = JengaBenchEnv()
        try:
            env.reset(seed=0)
            env.step({"type": "ChangeViewpoint", "azimuth": 90, "pitch": 5, "distance_cm": 60})
            result = env.step(
                {
                    "type": "Push",
                    "layer": 8,
                    "color": "Purple",
                    "face": "East",
                    "contact": "center",
                    "intensity": "Gentle",
                }
            )
            self.assertEqual(result.observation[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(env._consecutive_viewpoints, 0)
            self.assertEqual(result.info["outcome"], "settled")
            self.assertGreater(int(result.info["frame_count"]), 1)
            self.assertIsInstance(json.loads(result.info["replay_frames"]), list)
        finally:
            env.close()

    def test_invalid_model_push_returns_penalty(self) -> None:
        env = JengaBenchEnv()
        try:
            env.reset(seed=0)
            result = env.step(
                {
                    "type": "Push",
                    "layer": 8,
                    "color": "Purple",
                    "face": "North",
                    "contact": "center",
                    "intensity": "Gentle",
                }
            )
            self.assertEqual(result.reward, -0.5)
            self.assertFalse(result.terminated)
        finally:
            env.close()

    def test_model_extraction_requires_place_back_and_scores_point(self) -> None:
        env = JengaBenchEnv()
        try:
            env.reset(seed=0)
            result = env.step(
                {
                    "type": "Push",
                    "layer": 10,
                    "color": "Purple",
                    "face": "East",
                    "contact": "center",
                    "intensity": "Hard",
                }
            )
            self.assertFalse(result.terminated)
            self.assertEqual(result.reward, 1.0)
            self.assertEqual(result.info["outcome"], "extracted")
            self.assertEqual(result.info["phase"], "place_back")
            self.assertEqual(result.info["blocks_removed"], "1")
            self.assertIn("Available placement positions: Left, Middle, Right", result.system_prompt)
            placed = env.step({"type": "PlaceBack", "position": "Middle", "rotation_degrees": 0})
            self.assertFalse(placed.terminated)
            self.assertEqual(placed.info["outcome"], "placed")
            self.assertEqual(placed.info["phase"], "push")
        finally:
            env.close()


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class SandboxWebSocketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        preview.close()

    def test_websocket_streams_frames_and_result(self) -> None:
        with self.client.websocket_connect("/ws/sandbox") as websocket:
            websocket.send_json({"type": "Reset", "seed": 7})
            scene = websocket.receive_json()
            self.assertEqual(scene["type"], "scene")
            self.assertEqual(scene["scene"]["seed"], 7)
            websocket.send_json({"type": "Push", **REQUEST.__dict__})
            messages = []
            while True:
                message = websocket.receive_json()
                messages.append(message)
                if message["type"] == "result":
                    break
            frames = [message for message in messages if message["type"] == "frame"]
            self.assertGreater(len(frames), 1)
            self.assertEqual(messages[-1]["outcome"], "settled")

    def test_websocket_rejects_push_while_busy(self) -> None:
        self.assertTrue(motion_lock.acquire(blocking=False))
        try:
            with self.client.websocket_connect("/ws/sandbox") as websocket:
                websocket.send_json({"type": "Push", **REQUEST.__dict__})
                self.assertEqual(websocket.receive_json(), {"type": "error", "message": "busy"})
        finally:
            motion_lock.release()

    def test_preview_requires_reset_after_collapse(self) -> None:
        preview.reset_scene(seed=0)
        assert preview._simulation is not None
        target = preview._simulation._validate_push(REQUEST)
        other = next(block for block in preview._simulation.blocks if block.body_id != target.body_id)
        position, rotation = bullet.getBasePositionAndOrientation(
            other.body_id, physicsClientId=preview._simulation.client_id
        )
        bullet.resetBasePositionAndOrientation(
            other.body_id,
            (position[0], position[1], -0.01),
            rotation,
            physicsClientId=preview._simulation.client_id,
        )

        frames = preview.push(REQUEST)

        self.assertEqual(frames[-1]["phase"], "collapse")
        with self.assertRaisesRegex(RuntimeError, "reset is required"):
            preview.push(REQUEST)
        preview.reset_scene(seed=0)
        self.assertEqual(preview.push(REQUEST)[-1]["phase"], "settled")


if __name__ == "__main__":
    unittest.main()
