"""HTTP and OpenAI-compatible request helpers for NInfer.

The node intentionally uses the Python standard library for HTTP.  That keeps
the custom node usable in a stock ComfyUI environment and avoids maintaining
another client connection pool for a server that lives for one request only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener


class NInferClientError(RuntimeError):
    """Base class for errors returned by the local NInfer client."""


class NInferTimeoutError(NInferClientError, TimeoutError):
    """The health check or generation request exceeded its timeout."""


class NInferHTTPError(NInferClientError):
    """NInfer returned a non-success HTTP response."""

    def __init__(self, status: int, message: str, body: str = "") -> None:
        self.status = status
        self.body = body
        detail = f"HTTP {status}: {message}"
        if body:
            detail += f"; body={body[:500]}"
        super().__init__(detail)


class NInferProtocolError(NInferClientError):
    """The server returned JSON that does not match the expected contract."""


@dataclass(frozen=True)
class ChatRequest:
    """A non-streaming OpenAI Chat Completions request."""

    model: str
    messages: Sequence[Mapping[str, Any]]
    max_tokens: int = 192
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    reasoning_effort: str = "none"
    seed: int | None = None
    timeout_s: float = 180.0
    extra_payload: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Return a JSON-serializable request body.

        ``repetition_penalty`` is included only when it differs from the
        neutral value.  NInfer versions based on the documented OpenAI
        surface accept presence/frequency penalties, while some builds also
        accept this Qwen sampling extension.  Sending the extension only when
        requested keeps the default request compatible with both variants.
        """

        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.reasoning_effort not in {"none", "low", "medium"}:
            raise ValueError(
                "reasoning_effort must be one of: none, low, medium"
            )
        if not 0.0 <= self.temperature <= 2.0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and between 0 and 2")
        if not 0.0 <= self.top_p <= 1.0 or not math.isfinite(self.top_p):
            raise ValueError("top_p must be finite and between 0 and 1")
        if self.top_k < 0:
            raise ValueError("top_k must not be negative")
        if self.repetition_penalty <= 0 or not math.isfinite(self.repetition_penalty):
            raise ValueError("repetition_penalty must be positive and finite")
        if not math.isfinite(self.presence_penalty) or not math.isfinite(
            self.frequency_penalty
        ):
            raise ValueError("penalties must be finite")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative when present")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "reasoning_effort": self.reasoning_effort,
            "stream": False,
        }
        if self.repetition_penalty != 1.0:
            body["repetition_penalty"] = self.repetition_penalty
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(dict(self.extra_payload))
        return body


@dataclass(frozen=True)
class ChatResponse:
    """The assistant answer and the raw response envelope."""

    content: str
    reasoning_content: str | None
    raw: Mapping[str, Any]


def _clean_optional_text(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _build_user_text(
    user_prompt: str,
    negative_prompt: str,
    style_prompt: str,
    asset_description: str,
) -> str:
    sections: list[str] = []
    prompt = user_prompt.strip()
    if prompt:
        sections.append(prompt)
    style = style_prompt.strip()
    if style:
        sections.append(f"Style / enhancement guidance:\n{style}")
    negative = negative_prompt.strip()
    if negative:
        sections.append(
            "Negative prompt / exclusions (do not include these in the result):\n"
            f"{negative}"
        )
    asset = asset_description.strip()
    if asset:
        sections.append(f"Asset description:\n{asset}")
    return "\n\n".join(sections)


def build_chat_request(
    *,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    negative_prompt: str = "",
    style_prompt: str = "",
    asset_description: str = "",
    image_urls: Sequence[str] = (),
    reasoning_mode: str = "disabled",
    max_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    seed: int = -1,
    timeout_s: float = 180.0,
) -> ChatRequest:
    """Build the one-turn request used by the ComfyUI node."""

    if reasoning_mode not in {"disabled", "low", "medium"}:
        raise ValueError("reasoning_mode must be disabled, low, or medium")
    if any(not isinstance(url, str) or not url.startswith("data:") for url in image_urls):
        raise ValueError("image_urls must contain only data URLs")
    user_text = _build_user_text(
        user_prompt, negative_prompt, style_prompt, asset_description
    )
    if not user_text and not image_urls:
        raise ValueError("user_prompt or an image is required")

    if image_urls:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        ]
        if user_text:
            content.append({"type": "text", "text": user_text})
        user_message: dict[str, Any] = {"role": "user", "content": content}
    else:
        user_message = {"role": "user", "content": user_text}

    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append(user_message)

    effort = {"disabled": "none", "low": "low", "medium": "medium"}[reasoning_mode]
    return ChatRequest(
        model=model_id.strip(),
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        reasoning_effort=effort,
        seed=None if seed < 0 else seed,
        timeout_s=timeout_s,
    )


def parse_chat_response(payload: Mapping[str, Any]) -> ChatResponse:
    """Extract only ``choices[0].message.content``.

    NInfer returns reasoning separately as ``reasoning_content``.  It is
    deliberately retained as metadata but never concatenated with the answer
    returned to ComfyUI.
    """

    if not isinstance(payload, Mapping):
        raise NInferProtocolError("chat response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise NInferProtocolError("chat response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise NInferProtocolError("choices[0] must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise NInferProtocolError("choices[0].message must be an object")
    if "content" not in message:
        raise NInferProtocolError("choices[0].message.content is missing")
    content = message["content"]
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise NInferProtocolError("choices[0].message.content must be a string")
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        raise NInferProtocolError("reasoning_content must be a string when present")
    return ChatResponse(content=content, reasoning_content=reasoning, raw=payload)


@dataclass(frozen=True)
class HTTPResult:
    status: int
    body: bytes
    headers: Mapping[str, str]


class NInferClient:
    """Minimal JSON client for one NInfer server process."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        opener: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._opener = opener or build_opener()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> HTTPResult:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=timeout_s) as response:
                return HTTPResult(
                    status=int(response.status),
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
        except HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            raise NInferHTTPError(exc.code, str(exc.reason), error_body) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, OSError)) and (
                isinstance(reason, TimeoutError)
                or "timed out" in str(reason).lower()
            ):
                raise NInferTimeoutError(
                    f"NInfer request timed out after {timeout_s:.2f}s"
                ) from exc
            raise NInferClientError(f"NInfer connection failed: {reason}") from exc
        except TimeoutError as exc:
            # urllib can surface socket.timeout directly on some Python builds.
            raise NInferTimeoutError(
                f"NInfer request timed out after {timeout_s:.2f}s"
            ) from exc
        except OSError as exc:
            raise NInferClientError(f"NInfer connection failed: {exc}") from exc

    def get_json(self, path: str, *, timeout_s: float = 30.0) -> Mapping[str, Any]:
        result = self.request("GET", path, timeout_s=timeout_s)
        try:
            value = json.loads(result.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NInferProtocolError(f"{path} did not return JSON") from exc
        if not isinstance(value, Mapping):
            raise NInferProtocolError(f"{path} JSON response must be an object")
        return value

    def complete(self, request: ChatRequest) -> ChatResponse:
        result = self.request(
            "POST",
            "/v1/chat/completions",
            body=request.payload(),
            timeout_s=request.timeout_s,
        )
        try:
            value = json.loads(result.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NInferProtocolError("chat completion did not return JSON") from exc
        return parse_chat_response(value)
