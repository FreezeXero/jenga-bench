"""Your benchmark logic — implement reset() and step()."""

from __future__ import annotations

import random
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

ITEMS = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What color is the sky?", "answer": "blue"},
]


class MyEnv(BaseEnv):
    def __init__(self) -> None:
        self._item: dict[str, Any] | None = None
        self._rng = random.Random()

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._rng.seed(seed)
        self._item = self._rng.choice(ITEMS)
        return {"question": self._item["question"]}

    def step(self, action: Any) -> StepResult:
        if self._item is None:
            raise RuntimeError("Call reset() before step()")
        response = str(action).strip().lower()
        correct = response == self._item["answer"].lower()
        return StepResult(
            observation={"result": "done"},
            reward=1.0 if correct else 0.0,
            terminated=True,
            truncated=False,
            info={"correct": str(correct), "given_answer": response},
        )
