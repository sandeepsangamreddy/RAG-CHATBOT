"""Deterministic tests for the pre-generation refusal gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.config import settings
from src.generator import REFUSAL, AnswerGenerator


class MockMessages:
    """Track model calls and return one deterministic answer."""

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(
            type="text", text="The free look period is stated in the policy. [Section E.1.1, p.15]"
        )])


class MockClient:
    """Provide the messages interface used by the Anthropic SDK client."""

    def __init__(self) -> None:
        self.messages = MockMessages()


@pytest.fixture
def generator() -> tuple[AnswerGenerator, MockClient]:
    client = MockClient()
    instance = AnswerGenerator(settings, client=client)
    return instance, client


@pytest.mark.parametrize(
    "query",
    ["what's the weather in Delhi", "who is the CEO", "vet bills for my dog"],
)
def test_nonsense_queries_refuse_without_calling_llm(
    generator: tuple[AnswerGenerator, MockClient], query: str
) -> None:
    answer_generator, client = generator
    result = answer_generator.generate(query, [], retrieval_confidence=0.0)
    assert result["answer"] == REFUSAL
    assert result["refused"] is True
    assert client.messages.calls == 0


def test_in_scope_free_look_query_does_not_refuse(
    generator: tuple[AnswerGenerator, MockClient]
) -> None:
    answer_generator, client = generator
    result = answer_generator.generate(
        "what is the free look period",
        [],
        retrieval_confidence=0.5,
    )
    assert result["refused"] is False
    assert "free look period" in result["answer"].lower()
    assert client.messages.calls == 1
