"""Plain-SDK LLM clients used only by the evaluation harness.

The existing pipeline keeps using ``scripts/generate.py``. This module adds a
small, provider-agnostic JSON-completion helper for draft generation and
LLM-as-judge calls. No LangChain, no vendor lock-in beyond the optional SDKs.

Secrets are read from the project ``.env`` and are never printed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Load the project .env explicitly so the harness works from any directory.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # python-dotenv is a project dependency, but stay import-safe
    pass

from config import LLM_MODEL, LLM_PROVIDER  # noqa: E402

T = TypeVar("T", bound=BaseModel)

# Verified working defaults (2026-11). Override with --judge-model.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"

PROMPT_VERSION = "eval-v2"


class LLMConfigError(RuntimeError):
    """Missing key or unsupported provider."""


class LLMJsonError(RuntimeError):
    """The model did not return JSON matching the requested schema."""


@dataclass
class Usage:
    """Normalized token usage across providers."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
        }


@dataclass
class LLMResponse:
    """One non-streaming completion."""

    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""


@dataclass
class JsonResult:
    """A validated pydantic object plus the provider's usage metadata."""

    value: BaseModel
    usage: Usage
    raw: str
    model: str
    provider: str


# --------------------------------------------------------------------------- #
# Provider clients
# --------------------------------------------------------------------------- #
class LLMClient:
    """Minimal interface: complete(system, user, temperature, json_mode)."""

    provider = ""
    model = ""

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, json_mode: bool = True
    ) -> LLMResponse:
        raise NotImplementedError


class GroqClient(LLMClient):
    provider = "groq"

    def __init__(self, model: str, max_tokens: int = 2048) -> None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMConfigError(
                "GROQ_API_KEY is not set. Add it to .env or use --judge-provider gemini."
            )
        from groq import Groq

        self._client = Groq(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, json_mode: bool = True
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            usage=Usage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            model=self.model,
            provider=self.provider,
        )


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, max_tokens: int = 2048) -> None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMConfigError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self._client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, json_mode: bool = True
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            usage=Usage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            model=self.model,
            provider=self.provider,
        )


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, max_tokens: int = 2048) -> None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMConfigError("ANTHROPIC_API_KEY is not set.")
        import anthropic

        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, json_mode: bool = True
    ) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        usage = getattr(resp, "usage", None)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            usage=Usage(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            ),
            model=self.model,
            provider=self.provider,
        )


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str, max_tokens: int = 2048) -> None:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMConfigError(
                "GOOGLE_API_KEY / GEMINI_API_KEY is not set. "
                "Add it to .env or use --judge-provider groq."
            )
        from google import genai
        from google.genai import types

        self._types = types
        # v1beta enables JSON mode for gemini-2.5/3.x flash models.
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, json_mode: bool = True
    ) -> LLMResponse:
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )
        meta = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
        )
        return LLMResponse(
            text=resp.text or "",
            usage=usage,
            model=self.model,
            provider=self.provider,
        )


PROVIDER_DEFAULTS = {
    "gemini": DEFAULT_GEMINI_MODEL,
    "groq": DEFAULT_GROQ_MODEL,
    "openai": DEFAULT_OPENAI_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
}


def make_client(provider: str, model: str, *, max_tokens: int = 2048) -> LLMClient:
    """Build a provider client from a short provider name."""
    provider = (provider or "").strip().lower()
    if provider == "gemini":
        return GeminiClient(model, max_tokens=max_tokens)
    if provider == "groq":
        return GroqClient(model, max_tokens=max_tokens)
    if provider in ("openai", "openai-compatible"):
        return OpenAIClient(model, max_tokens=max_tokens)
    if provider == "anthropic":
        return AnthropicClient(model, max_tokens=max_tokens)
    raise LLMConfigError(
        f"Unsupported provider {provider!r}; use gemini, groq, openai or anthropic."
    )


def provider_status() -> dict[str, bool]:
    """Which provider keys are present (booleans only, never key values)."""
    return {
        "gemini": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


def resolve_judge_defaults(
    provider: str | None = None, model: str | None = None
) -> tuple[str, str]:
    """Pick a judge provider/model, preferring one different from the generator.

    Groq comes first because it is fast enough for full runs. Gemini works but
    is rate-limited and can take minutes per question on the free tier.
    """
    if not provider:
        if provider_status()["groq"]:
            provider = "groq"
        elif provider_status()["gemini"]:
            provider = "gemini"
        elif provider_status()["openai"]:
            provider = "openai"
        elif provider_status()["anthropic"]:
            provider = "anthropic"
        else:
            raise LLMConfigError(
                "No judge API key found. Set GOOGLE_API_KEY, GROQ_API_KEY, "
                "OPENAI_API_KEY or ANTHROPIC_API_KEY."
            )
    provider = provider.lower()
    model = model or PROVIDER_DEFAULTS.get(provider)
    if not model:
        raise LLMConfigError(f"No default model for provider {provider!r}.")
    if provider == (LLM_PROVIDER or "").lower() and model == LLM_MODEL:
        print(
            "  WARNING: judge and generator use the same model "
            f"({provider}/{model}); judges tend to favor their own answers.",
            file=sys.stderr,
        )
    return provider, model


def generator_spec() -> dict[str, str]:
    """Generator provider/model from the existing pipeline config."""
    return {"provider": (LLM_PROVIDER or "").lower(), "model": LLM_MODEL}


# --------------------------------------------------------------------------- #
# Retry / JSON parsing
# --------------------------------------------------------------------------- #
def is_retryable(exc: BaseException) -> bool:
    """Rate limit, transient server and network failures are worth retrying."""
    for attr in ("status_code", "code"):
        if getattr(exc, attr, None) in (429, 500, 502, 503, 504):
            return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "ratelimit",
        "rate limit",
        "timeout",
        "connection",
        "apiconnection",
        "internalserver",
        "serviceunavailable",
        "overloaded",
        "try again",
        "resource_exhausted",
    )
    return any(marker in name or marker in text for marker in markers)


def _retry_after(exc: BaseException) -> float | None:
    """Best-effort Retry-After extraction."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_seconds: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    label: str = "LLM",
) -> T:
    """Call ``fn`` and retry retryable failures with exponential backoff."""
    last: BaseException | None = None
    for attempt in range(max(1, retries)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classify, then re-raise
            if not is_retryable(exc):
                raise
            last = exc
            wait = _retry_after(exc) or (base_seconds * (2**attempt))
            print(
                f"  {label} retry {attempt + 1}/{max(1, retries) - 1} "
                f"in {wait:.1f}s ({type(exc).__name__})",
                file=sys.stderr,
            )
            sleep_fn(wait)
    raise RuntimeError(f"{label} still failing after retries: {last}") from last


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output.

    Handles markdown fences and reasoning text before the JSON object.
    """
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found in model output")


def complete_json(
    client: LLMClient,
    system: str,
    user: str,
    schema: type[T],
    *,
    temperature: float = 0.0,
    max_parse_retries: int = 1,
    retries: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
    label: str = "LLM",
) -> JsonResult:
    """Call the model, extract JSON, validate with pydantic, retry once on bad JSON."""
    prompt = user
    last_error: BaseException | None = None
    for attempt in range(max_parse_retries + 1):
        response = retry_call(
            lambda: client.complete(
                system, prompt, temperature=temperature, json_mode=True
            ),
            retries=retries,
            sleep_fn=sleep_fn,
            label=label,
        )
        try:
            data = extract_json(response.text)
            value = schema.model_validate(data)
            return JsonResult(
                value=value,
                usage=response.usage,
                raw=response.text,
                model=response.model,
                provider=response.provider,
            )
        except (ValueError, ValidationError) as exc:
            last_error = exc
            if attempt < max_parse_retries:
                prompt = (
                    f"{user}\n\nIMPORTANT: your previous response was not valid JSON "
                    f"for the required schema ({exc}). Return ONLY a valid JSON object."
                )
                print(
                    f"  {label} JSON parse failed; retrying once",
                    file=sys.stderr,
                )
    raise LLMJsonError(
        f"{label} did not return valid JSON after retries: {last_error}"
    ) from last_error


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token)."""
    return max(1, int((len(text or "") + 3) // 4))


def estimate_call_tokens(*texts: str, output_tokens: int = 400) -> int:
    """Estimated input + output tokens for one JSON call."""
    return sum(estimate_tokens(text) for text in texts) + output_tokens
