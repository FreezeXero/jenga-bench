from __future__ import annotations

import importlib.util
import json
import unittest

PHYSICS_AVAILABLE = bool(importlib.util.find_spec("pybullet"))

if PHYSICS_AVAILABLE:
    from fastapi.testclient import TestClient

    from env import JengaBenchEnv
    from showcase.server import _normalize_replay, app, preview


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class EnvironmentReplayTraceTests(unittest.TestCase):
    def test_step_info_contains_cumulative_recorded_replay(self) -> None:
        env = JengaBenchEnv()
        try:
            env.reset(seed=3)
            first = env.step(
                {
                    "type": "ChangeViewpoint",
                    "context": "Inspect the east face.",
                    "direction": "E",
                    "elevation_layer": 9,
                    "distance": "Medium",
                }
            )
            second = env.step(
                {
                    "type": "Push",
                    "context": "Probe a middle block gently.",
                    "layer": 8,
                    "color": "Green",
                    "face": "East",
                    "contact": "center",
                    "intensity": "Gentle",
                }
            )

            first_trace = json.loads(first.info["episode_replay"])
            trace = json.loads(second.info["episode_replay"])
            self.assertEqual(second.info["replay_schema_version"], "1")
            self.assertIsNotNone(trace["initial_frame"])
            self.assertEqual(len(first_trace["steps"]), 1)
            self.assertEqual(len(trace["steps"]), 2)
            self.assertTrue(trace["steps"][0]["agent_frame"].startswith("iVBOR"))
            self.assertEqual(trace["steps"][0]["physics_frames"], [])
            self.assertEqual(trace["steps"][1]["see"], "")
            self.assertEqual(trace["steps"][1]["do"], "")
            self.assertEqual(trace["steps"][1]["next"], "")
            self.assertEqual(trace["steps"][1]["action"]["context"], "Probe a middle block gently.")
            self.assertGreater(len(trace["steps"][1]["physics_frames"]), 1)
        finally:
            env.close()

    def test_invalid_action_is_recorded(self) -> None:
        env = JengaBenchEnv()
        try:
            env.reset(seed=0)
            result = env.step({"type": "Push"})
            step = json.loads(result.info["episode_replay"])["steps"][0]
            self.assertEqual(step["action"], {"type": "Push"})
            self.assertTrue(step["events"][0].startswith("invalid_action:"))
        finally:
            env.close()


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires pybullet; run in Dockerfile.physics")
class ShowcaseReplayApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        preview.close()

    def test_catalog_and_legacy_normalization(self) -> None:
        catalog = self.client.get("/api/replays")
        self.assertEqual(catalog.status_code, 200)
        self.assertGreater(len(catalog.json()), 0)
        payload = {
            "run": {"id": "legacy", "scores": {"normalized_score": 3.06}},
            "episodes": [
                {
                    "id": "episode",
                    "steps": 3,
                    "terminal_info": {
                        "blocks_removed": "3",
                        "replay_frames": json.dumps([{"blocks": []}, {"blocks": []}]),
                    },
                }
            ],
        }
        replay = _normalize_replay(payload)
        episode = replay["episodes"][0]
        self.assertEqual(episode["completeness"], "partial")
        self.assertEqual(len(episode["steps"]), 3)
        self.assertTrue(all(step["recording_status"] == "unavailable" for step in episode["steps"][:2]))
        self.assertEqual(episode["steps"][2]["recording_status"], "recorded")
        self.assertGreater(len(episode["steps"][2]["physics_frames"]), 1)
        self.assertEqual(episode["steps"][2]["action"], {})
        self.assertEqual(replay["successful_extractions"], 3)

    def test_unknown_replay_is_rejected(self) -> None:
        self.assertEqual(self.client.get("/api/replays/not-a-run").status_code, 404)

    def test_malformed_optional_json_degrades_to_empty_values(self) -> None:
        payload = {
            "run": {"id": "broken", "scores": {"normalized_score": 0}},
            "episodes": [{"id": "episode", "steps": 1, "terminal_info": {"replay_frames": "{"}}],
        }
        episode = _normalize_replay(payload)["episodes"][0]
        self.assertEqual(episode["completeness"], "partial")
        self.assertEqual(episode["steps"][0]["physics_frames"], [])

    def test_turn_shaped_trace_reads_nested_info(self) -> None:
        payload = {
            "run": {"id": "future", "scores": {"normalized_score": 0}},
            "episodes": [{"id": "episode", "terminal_info": {}}],
            "traces": {
                "episode": [
                    {
                        "action": {"type": "ChangeViewpoint"},
                        "reasoning": "Inspect from the east.",
                        "observation": {"data": "png"},
                        "info": {"replay_frames": "[]", "camera_state": "{\"direction\":\"E\"}"},
                    }
                ]
            },
        }
        step = _normalize_replay(payload)["episodes"][0]["steps"][0]
        self.assertEqual(step["recording_status"], "recorded")
        self.assertEqual(step["context"], "Inspect from the east.")
        self.assertEqual(step["agent_frame"], "png")
        self.assertEqual(step["camera_state"], {"direction": "E"})

    def test_benchmark_markup_has_replay_and_comparison_tabs(self) -> None:
        index = self.client.get("/").text
        self.assertIn("showBenchmarkTab('replay'", index)
        self.assertIn("showBenchmarkTab('comparison'", index)
        self.assertIn('id="replay-canvas"', index)
        self.assertIn('id="replay-canvas" width="620" height="620"', index)
        self.assertIn('id="replay-step-slider"', index)
        self.assertIn("class ReadOnlyReplayRenderer", index)
        self.assertIn("createShadowFramebuffer", index)
        self.assertIn("startStepPlayback", index)
        self.assertIn("recordedRenderer.resize(); recordedRenderer.render();", index)
        self.assertIn("function hydrateReplayFrame", index)
        self.assertIn("staticObservation=type==='ChangeViewpoint'", index)
        self.assertNotIn("Full Game Replay", index)
        self.assertNotIn("class RecordedTowerRenderer", index)
        self.assertNotIn("scrubReplay", index)
        self.assertIn("Partial replay", index)
        self.assertIn('id="replay-live-score"', index)
        self.assertIn('id="stat-run-extractions"', index)
        self.assertIn("Playable Projection", index)
        self.assertIn("AI Step State", index)
        self.assertIn("/static/icons.svg#", index)
        self.assertIn('<option value="0.25">0.25×</option>', index)
        self.assertIn('<option value="0.5">0.5×</option>', index)
        self.assertNotIn('<option value="250">4×</option>', index)


if __name__ == "__main__":
    unittest.main()
