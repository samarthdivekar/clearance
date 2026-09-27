"""LLM clients.

`AnthropicLLM` calls the Claude API. `FakeLLM` is a deterministic, extractive stand-in with the
same interface: it lets the full pipeline, the tests, the red-team suite and the cache
benchmarks run offline and for free. Token counts from FakeLLM are estimates (chars / 4) priced
at the real model's rates, so cost comparisons stay meaningful.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Protocol

from clearance.llm.pricing import cost_usd
from clearance.llm.prompts import NOT_FOUND

# Models that accept adaptive thinking + effort + server-side refusal fallbacks.
_ADAPTIVE_MODELS = {"claude-opus-5", "claude-opus-5-5", "claude-fable-5-1", "claude-opus-4-8", "claude-sonnet-5"}
_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    stop_reason: str = "end_turn"

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens, self.cache_read_tokens)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(
        self, system: str, user: str, model: str, max_tokens: int = 16000, effort: str | None = None
    ) -> LLMResult: ...

    def complete_json(self, system: str, user: str, model: str, schema: dict) -> tuple[dict, LLMResult]: ...


class AnthropicLLM:
    def __init__(self):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=3)

    def _request_kwargs(self, model: str, effort: str | None) -> dict:
        kwargs: dict = {}
        if model in _ADAPTIVE_MODELS:
            kwargs["thinking"] = {"type": "adaptive"}
            if effort:
                kwargs["output_config"] = {"effort": effort}
        return kwargs

    def _call(self, model: str, **kwargs):
        a = self._anthropic
        try:
            if model in _FALLBACK_MODELS:
                # On a safety-classifier refusal, the API re-runs the request on the model
                # Anthropic recommends for that category, inside the same call.
                return self.client.beta.messages.create(
                    model=model, betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            return self.client.messages.create(model=model, **kwargs)
        except a.AuthenticationError as e:
            raise LLMError("Anthropic authentication failed: set ANTHROPIC_API_KEY (or use CLEARANCE_LLM_PROVIDER=fake)") from e
        except a.BadRequestError as e:
            raise LLMError(f"Bad request to {model}: {e.message}") from e
        except a.RateLimitError as e:
            raise LLMError("Rate limited by the Claude API; retry later") from e
        except a.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError("Could not reach the Claude API") from e

    @staticmethod
    def _to_result(response, requested_model: str) -> LLMResult:
        if response.stop_reason == "refusal":
            text = "The model declined to answer this request."
        else:
            text = "".join(b.text for b in response.content if b.type == "text").strip()
        usage = response.usage
        return LLMResult(
            text=text,
            model=getattr(response, "model", None) or requested_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            stop_reason=response.stop_reason or "",
        )

    def complete(self, system, user, model, max_tokens=16000, effort=None) -> LLMResult:
        response = self._call(
            model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **self._request_kwargs(model, effort),
        )
        return self._to_result(response, model)

    def complete_json(self, system, user, model, schema) -> tuple[dict, LLMResult]:
        kwargs = self._request_kwargs(model, None)
        output_config = kwargs.pop("output_config", {})
        output_config["format"] = {"type": "json_schema", "schema": schema}
        response = self._call(
            model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            **kwargs,
        )
        result = self._to_result(response, model)
        if response.stop_reason == "refusal":
            raise LLMError("Model refused the structured-output request")
        return json.loads(result.text), result


_CONTEXT_RE = re.compile(r'<email n="(\d+)">\n(.*?)\n</email>', re.S)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of to and in on for is was who what when where why how did does do about with from".split())


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FakeLLM:
    """Extractive answerer: returns the context sentences that best overlap the question."""

    def complete(self, system, user, model, max_tokens=16000, effort=None) -> LLMResult:
        question = user.rsplit("Question:", 1)[-1]
        q_words = {w for w in _WORD_RE.findall(question.lower()) if w not in _STOP}
        min_overlap = max(2, math.ceil(0.3 * len(q_words)))
        best: list[tuple[float, int, str]] = []
        for n, body in _CONTEXT_RE.findall(user):
            text = body.split("\n", 1)[1] if "\n" in body else body
            for sent in re.split(r"(?<=[.!?])\s+", text):
                words = set(_WORD_RE.findall(sent.lower()))
                overlap = len(q_words & words)
                if overlap >= min(min_overlap, len(q_words)):
                    best.append((overlap / (1 + len(words) ** 0.5), int(n), sent.strip()))
        best.sort(key=lambda t: -t[0])
        if not best:
            answer = NOT_FOUND
        else:
            answer = " ".join(f"{s} [{n}]" for _, n, s in best[:2])
        return LLMResult(
            text=answer,
            model=model,
            input_tokens=_estimate_tokens(system + user),
            output_tokens=_estimate_tokens(answer),
        )

    def complete_json(self, system, user, model, schema) -> tuple[dict, LLMResult]:
        data = _fake_for_schema(schema)
        return data, LLMResult(json.dumps(data), model, _estimate_tokens(user), 20)


def _fake_for_schema(schema: dict):
    t = schema.get("type")
    if t == "object":
        return {k: _fake_for_schema(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return []
    if t == "number":
        return 1.0
    if t == "integer":
        return 1
    if t == "boolean":
        return True
    return ""


def load_llm(provider: str) -> LLM:
    if provider == "fake":
        return FakeLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"Unknown LLM provider {provider!r} (expected 'anthropic' or 'fake')")
