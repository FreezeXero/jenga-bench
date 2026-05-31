from __future__ import annotations

import importlib.util
import struct
import unittest

PHYSICS_AVAILABLE = bool(importlib.util.find_spec("pybullet"))

if PHYSICS_AVAILABLE:
    from fastapi.testclient import TestClient

    from env import JengaBenchEnv
    from jenga.render import IMAGE_HEIGHT, IMAGE_WIDTH, CameraPose, render_png
    from showcase.server import DEFAULT_CAMERA, PreviewState, app, preview


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class InspectorStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = PreviewState()

    def tearDown(self) -> None:
        self.state.close()

    def test_reset_returns_valid_png_and_default_camera(self) -> None:
        image, camera = self.state.reset()

        self.assertEqual(image[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", image[16:24]), (IMAGE_WIDTH, IMAGE_HEIGHT))
        self.assertEqual(camera, DEFAULT_CAMERA)

    def test_camera_changes_png_without_moving_tower(self) -> None:
        first, _ = self.state.reset()
        transforms = self.state.transforms()
        second, _ = self.state.frame(CameraPose(azimuth=90, pitch=5, distance_cm=60))

        self.assertNotEqual(first, second)
        self.assertEqual(transforms, self.state.transforms())

    def test_identical_camera_pose_returns_identical_png(self) -> None:
        self.state.reset()
        camera = CameraPose(azimuth=90, pitch=5, distance_cm=60)

        first, _ = self.state.frame(camera)
        second, _ = self.state.frame(camera)

        self.assertEqual(first, second)

    def test_reset_restores_default_camera_and_tower(self) -> None:
        first, _ = self.state.reset()
        transforms = self.state.transforms()
        self.state.frame(CameraPose(azimuth=90, pitch=5, distance_cm=60))

        second, camera = self.state.reset()

        self.assertEqual(camera, DEFAULT_CAMERA)
        self.assertEqual(first, second)
        self.assertEqual(transforms, self.state.transforms())

    def test_reset_seed_replays_and_varies_tower(self) -> None:
        first, _ = self.state.reset(seed=7)
        transforms = self.state.transforms()
        replay, _ = self.state.reset(seed=7)
        self.assertEqual(first, replay)
        self.assertEqual(transforms, self.state.transforms())
        varied, _ = self.state.reset(seed=8)
        self.assertNotEqual(first, varied)


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class InspectorRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        preview.close()

    def test_routes_and_static_assets_load(self) -> None:
        index = self.client.get("/")
        script = self.client.get("/static/app.js")
        styles = self.client.get("/static/styles.css")
        health = self.client.get("/health")

        self.assertEqual(index.status_code, 200)
        self.assertIn("Live Tower Inspector", index.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("renderScene", script.text)
        self.assertNotIn('fetch("/api/frame"', script.text)
        self.assertIn("scene.target[0] + Math.cos(pitch) * Math.sin(yaw)", script.text)
        self.assertIn("scene.target[1] - Math.cos(pitch) * Math.cos(yaw)", script.text)
        self.assertIn("gl.enable(gl.DEPTH_TEST)", script.text)
        self.assertIn("camera.azimuth - deltaX", script.text)
        self.assertIn("new WebSocket", script.text)
        self.assertIn("function slerp", script.text)
        self.assertIn("Push block", index.text)
        self.assertIn("Place block on top", index.text)
        self.assertIn('type: "PlaceBack"', script.text)
        self.assertIn("applyScene(message.scene, { preserveCamera: true })", script.text)
        self.assertIn("function playQueuedFrames()", script.text)
        self.assertIn("pendingResult = message", script.text)
        self.assertIn("function framesVisuallyMatch", script.text)
        self.assertIn("clamp(camera.pitch + deltaY * .4, -45, 75)", script.text)
        self.assertIn('sandboxTerminated = message.outcome === "collapse"', script.text)
        self.assertEqual(script.headers["cache-control"], "no-store")
        self.assertEqual(styles.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

    def test_state_returns_local_render_scene(self) -> None:
        response = self.client.get("/api/state")

        self.assertEqual(response.status_code, 200)
        scene = response.json()
        self.assertEqual(scene["camera"]["azimuth"], 225.0)
        self.assertEqual(len(scene["blocks"]), 54)
        self.assertEqual(scene["base"]["size"], [0.25, 0.25, 0.045])

    def test_capture_returns_png_with_camera_headers(self) -> None:
        response = self.client.post(
            "/api/capture",
            json={"azimuth": 90, "pitch": 5, "distance_cm": 60},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-camera-azimuth"], "90.00")
        self.assertEqual(struct.unpack(">II", response.content[16:24]), (512, 512))

    def test_reset_returns_local_render_scene(self) -> None:
        response = self.client.post("/api/reset?seed=7")

        self.assertEqual(response.status_code, 200)
        scene = response.json()
        self.assertEqual(scene["camera"]["azimuth"], 225.0)
        self.assertEqual(scene["seed"], 7)
        self.assertEqual(len(scene["blocks"]), 54)

    def test_invalid_camera_values_are_rejected(self) -> None:
        cases = [
            {"azimuth": -1, "pitch": 5, "distance_cm": 60},
            {"azimuth": 90, "pitch": -46, "distance_cm": 60},
            {"azimuth": 90, "pitch": 76, "distance_cm": 60},
            {"azimuth": 90, "pitch": 5, "distance_cm": 121},
        ]

        for camera in cases:
            with self.subTest(camera=camera):
                response = self.client.post("/api/capture", json=camera)
                self.assertEqual(response.status_code, 422)

    def test_negative_pitch_is_allowed(self) -> None:
        response = self.client.post(
            "/api/capture",
            json={"azimuth": 90, "pitch": -45, "distance_cm": 60},
        )

        self.assertEqual(response.status_code, 200)


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class RenderCompatibilityTests(unittest.TestCase):
    def test_environment_observation_matches_direct_current_camera_render(self) -> None:
        env = JengaBenchEnv()
        try:
            env.reset(seed=3)
            result = env.step(
                {"type": "ChangeViewpoint", "azimuth": 90, "pitch": 5, "distance_cm": 60}
            )
            direct = render_png(
                env._simulation,
                CameraPose(azimuth=90, pitch=5, distance_cm=60),
            )

            self.assertEqual(result.observation, direct)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
