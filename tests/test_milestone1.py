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


def viewpoint(direction: str = "NE", elevation_layer: int = 9, distance: str = "Full") -> dict:
    return {
        "type": "ChangeViewpoint",
        "direction": direction,
        "elevation_layer": elevation_layer,
        "distance": distance,
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
        result = env.step(viewpoint(direction="E", elevation_layer=5, distance="Medium"))

        self.assertEqual(result.content_type, PNG_CONTENT_TYPE)
        self.assertNotEqual(initial["data"], result.observation)
        self.assertIn("direction=E", result.system_prompt or "")
        self.assertIn("elevation_layer=5", result.system_prompt or "")
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.terminated)

    def test_same_seed_and_actions_are_byte_identical(self) -> None:
        actions = [viewpoint("NE"), viewpoint("SE", elevation_layer=5), viewpoint("W")]
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
            result = env.step(viewpoint(direction=["N","NE","E","SE","S","SW","W","NW","N"][index]))
            self.assertFalse(result.terminated)

        result = env.step(viewpoint(direction="S"))

        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, -10.0)
        self.assertEqual(result.info["raw_points"], "-10.00")
        self.assertEqual(result.info["normalized_score"], "-10.20")
        self.assertEqual(result.info["termination_reason"], "viewpoint_timeout")

    def test_change_viewpoint_without_target_block_leaves_camera_untargeted(self) -> None:
        env = JengaBenchEnv()
        env.reset(seed=1)

        result = env.step(viewpoint(direction="E", elevation_layer=9, distance="Medium"))

        self.assertNotIn("target_block", json.loads(result.info["camera_state"]))

    def test_targeted_viewpoint_can_be_cleared_by_omitting_target_block(self) -> None:
        env = JengaBenchEnv()
        env.reset(seed=1)

        first = env.step(
            {
                "type": "ChangeViewpoint",
                "direction": "E",
                "elevation_layer": 9,
                "distance": "Medium",
                "target_block": {"layer": 10, "color": "Green"},
            }
        )
        second = env.step(viewpoint(direction="E", elevation_layer=9, distance="Medium"))

        self.assertIn("target_block", json.loads(first.info["camera_state"]))
        self.assertNotIn("target_block", json.loads(second.info["camera_state"]))

    def test_change_viewpoint_accepts_green_and_rejects_brown_target(self) -> None:
        env = JengaBenchEnv()
        env.reset(seed=1)

        accepted = env.step(
            {
                "type": "ChangeViewpoint",
                "direction": "E",
                "elevation_layer": 9,
                "distance": "Medium",
                "target_block": {"layer": 10, "color": "Green"},
            }
        )
        rejected = env.step(
            {
                "type": "ChangeViewpoint",
                "direction": "E",
                "elevation_layer": 9,
                "distance": "Medium",
                "target_block": {"layer": 10, "color": "Brown"},
            }
        )

        self.assertEqual(accepted.reward, 0.0)
        self.assertEqual(rejected.reward, INVALID_ACTION_PENALTY)
        self.assertIn("Blue, Green, or Red", rejected.info["events"])


class ManifestTests(unittest.TestCase):
    def test_manifest_validates(self) -> None:
        domain = domain_config_from_manifest(ROOT / "benchanything.json")

        self.assertEqual(domain.id, "jenga-bench")
        self.assertEqual(domain.binding_vow.observation_space.type, SpaceType.IMAGE)
        self.assertEqual(domain.scoring.primary_metric, "normalized_score")

    def test_manifest_is_json(self) -> None:
        manifest = json.loads((ROOT / "benchanything.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tags"], ["tier1"])
        self.assertEqual(manifest["binding_vow"]["action_space"]["type"], "json")
        action_space = manifest["binding_vow"]["action_space"]
        self.assertIn("PlaceBack", action_space["description"])

        schema = json.loads(action_space["schema_ref"])
        variants = {variant["properties"]["type"]["const"]: variant for variant in schema["oneOf"]}
        self.assertEqual(set(variants), {"ChangeViewpoint", "Push", "PlaceBack"})
        self.assertIn("direction", variants["ChangeViewpoint"]["properties"])
        self.assertIn("elevation_layer", variants["ChangeViewpoint"]["properties"])
        self.assertIn("distance", variants["ChangeViewpoint"]["properties"])
        self.assertEqual(variants["Push"]["properties"]["layer"]["minimum"], 1)
        self.assertEqual(variants["Push"]["properties"]["intensity"]["enum"], ["Gentle", "Firm", "Hard"])
        self.assertIn("position", variants["PlaceBack"]["properties"])
        self.assertEqual(
            variants["Push"]["properties"]["color"]["enum"],
            ["Blue", "Green", "Red"],
        )
        self.assertEqual(
            variants["ChangeViewpoint"]["properties"]["target_block"]["properties"]["color"]["enum"],
            ["Blue", "Green", "Red"],
        )
        self.assertTrue(all(variant["additionalProperties"] is False for variant in variants.values()))


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
