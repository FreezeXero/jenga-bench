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
        PushRequest,
        PushValidationError,
    )
    from showcase.server import app, motion_lock, preview


REQUEST = PushRequest(layer=8, color="Purple", face="East", contact="center", intensity="Gentle")


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
            second.reset(seed=999)
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

    def test_force_levels_are_locked(self) -> None:
        self.assertEqual(INTENSITIES, {"Gentle": 0.15, "Firm": 0.60, "Hard": 1.20})

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
            websocket.send_json({"type": "Reset"})
            self.assertEqual(websocket.receive_json()["type"], "scene")
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


if __name__ == "__main__":
    unittest.main()
