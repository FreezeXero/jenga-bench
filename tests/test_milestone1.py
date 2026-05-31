from __future__ import annotations

import inspect
import importlib.util
import json
import struct
import unittest
from pathlib import Path

from bench_common.core.binding_vow import SpaceType
from bench_common.env_sdk.base import StepResult
from bench_common.env_sdk.manifest import domain_config_from_manifest
from bench_common.runtime import inference

from env import (
    INVALID_ACTION_PENALTY,
    PNG_CONTENT_TYPE,
    VIEWPOINT_LIMIT,
    JengaBenchEnv,
)

ROOT = Path(__file__).resolve().parents[1]
PHYSICS_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "pybullet"))


def viewpoint(azimuth: float = 45.0, pitch: float = 12.0, distance_cm: float = 50.0) -> dict:
    return {
        "type": "ChangeViewpoint",
        "azimuth": azimuth,
        "pitch": pitch,
        "distance_cm": distance_cm,
    }


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires numpy and pybullet; run in Dockerfile.physics")
class PngContractTests(unittest.TestCase):
    def test_reset_returns_valid_png_bytes(self) -> None:
        observation = JengaBenchEnv().reset(seed=7)
        data = observation["data"]

        self.assertEqual(observation["content_type"], PNG_CONTENT_TYPE)
        self.assertIsInstance(data, bytes)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        self.assertEqual((width, height), (512, 512))

    def test_viewpoint_updates_png_and_prompt(self) -> None:
        env = JengaBenchEnv()
        initial = env.reset(seed=7)
        result = env.step(viewpoint(azimuth=90.0, pitch=-5.0, distance_cm=70.0))

        self.assertEqual(result.content_type, PNG_CONTENT_TYPE)
        self.assertNotEqual(initial["data"], result.observation)
        self.assertIn("azimuth=90.00", result.system_prompt or "")
        self.assertIn("pitch=-5.00", result.system_prompt or "")
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.terminated)

    def test_same_seed_and_actions_are_byte_identical(self) -> None:
        actions = [viewpoint(20.0), viewpoint(140.0, pitch=30.0), viewpoint(280.0)]
        first = JengaBenchEnv()
        second = JengaBenchEnv()

        self.assertEqual(first.reset(seed=19), second.reset(seed=19))
        for action in actions:
            left = first.step(action)
            right = second.step(action)
            self.assertEqual(left.observation, right.observation)
            self.assertEqual(left.reward, right.reward)
            self.assertEqual(left.terminated, right.terminated)
            self.assertEqual(left.info, right.info)


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires numpy and pybullet; run in Dockerfile.physics")
class ActionContractTests(unittest.TestCase):
    def test_invalid_action_returns_penalty_result(self) -> None:
        env = JengaBenchEnv()
        env.reset(seed=1)

        result = env.step({"type": "Push"})

        self.assertEqual(result.reward, INVALID_ACTION_PENALTY)
        self.assertFalse(result.terminated)
        self.assertIn("invalid_action", result.info["events"])
        self.assertEqual(result.info["raw_points"], "-0.50")

    def test_tenth_viewpoint_terminates_with_negative_normalized_score(self) -> None:
        env = JengaBenchEnv()
        env.reset(seed=1)

        for index in range(VIEWPOINT_LIMIT - 1):
            result = env.step(viewpoint(azimuth=float(index * 10)))
            self.assertFalse(result.terminated)

        result = env.step(viewpoint(azimuth=180.0))

        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, -10.0)
        self.assertEqual(result.info["raw_points"], "-10.00")
        self.assertEqual(result.info["normalized_score"], "-10.20")
        self.assertEqual(result.info["termination_reason"], "viewpoint_timeout")


class ManifestTests(unittest.TestCase):
    def test_manifest_validates(self) -> None:
        domain = domain_config_from_manifest(ROOT / "benchanything.json")

        self.assertEqual(domain.id, "jenga-bench")
        self.assertEqual(domain.binding_vow.observation_space.type, SpaceType.IMAGE)
        self.assertEqual(domain.scoring.primary_metric, "normalized_score")

    def test_manifest_is_json(self) -> None:
        manifest = json.loads((ROOT / "benchanything.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["binding_vow"]["action_space"]["type"], "json")


class PinnedSdkCapabilityTests(unittest.TestCase):
    def test_sdk_has_image_observation_pipeline(self) -> None:
        self.assertIn("content_type", StepResult.__dataclass_fields__)
        source = inspect.getsource(inference.InferenceRouter)
        self.assertIn("_build_image_content", source)
        self.assertIn("_build_user_content", source)

        router = inference.InferenceRouter(allow_any_model=True)
        openai_blocks = router._build_image_content(b"png", "image/png", "[Step 1]", "openai")
        anthropic_blocks = router._build_image_content(
            b"png", "image/png", "[Step 1]", "anthropic"
        )
        self.assertEqual(openai_blocks[1]["type"], "image_url")
        self.assertTrue(openai_blocks[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(anthropic_blocks[1]["source"]["type"], "base64")
        self.assertEqual(anthropic_blocks[1]["source"]["media_type"], "image/png")


if __name__ == "__main__":
    unittest.main()
