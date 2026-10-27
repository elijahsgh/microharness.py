"""Lifecycle hooks demonstrating incoming (`--DERPDERPEICAR--`) and outgoing (`--DERPDERPEICAROUTGOING--`) guards."""

from __future__ import annotations

from collections.abc import AsyncIterable, Sequence
from dataclasses import replace
from typing import Any

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import ToolDefinition

INCOMING_FORBIDDEN_SIGNATURE = "--DERPDERPEICAR--"
OUTGOING_FORBIDDEN_SIGNATURE = "--DERPDERPEICAROUTGOING--"

INCOMING_REFUSAL_MESSAGE = (
    f"Refused: Prompt blocked by incoming hook `block_incoming_derp_eicar` because it "
    f"contains the forbidden `{INCOMING_FORBIDDEN_SIGNATURE}` signature. "
    "The request was not sent to the model."
)
OUTGOING_REFUSAL_MESSAGE = (
    "Refused: Outgoing response blocked by outgoing hook `block_outgoing_derp_eicar` because it "
    "contained the forbidden outgoing EICAR signature."
)

hooks = Hooks()


def _extract_user_prompt_text(part: UserPromptPart) -> str:
    """Extract text from a UserPromptPart (`str` in CLI query mode, `Sequence[UserContent]` in ACP mode)."""
    if isinstance(part.content, str):
        return part.content
    if isinstance(part.content, Sequence):
        return "\n".join(item for item in part.content if isinstance(item, str))
    return str(part.content)


def _redact_user_prompt_part(part: UserPromptPart, signature: str) -> None:
    """Redact `signature` in-place from a historical UserPromptPart."""
    if isinstance(part.content, str):
        if signature in part.content:
            part.content = part.content.replace(signature, "[REDACTED]")
    elif isinstance(part.content, Sequence) and any(
        isinstance(item, str) and signature in item for item in part.content
    ):
        part.content = [
            item.replace(signature, "[REDACTED]") if isinstance(item, str) else item
            for item in part.content
        ]


def _longest_prefix_suffix_len(text: str, signature: str) -> int:
    """Return the length of the longest suffix of `text` that is a non-empty proper prefix of `signature`."""
    max_len = min(len(text), len(signature) - 1)
    for length in range(max_len, 0, -1):
        if text.endswith(signature[:length]):
            return length
    return 0


def _response_contains_outgoing_forbidden(response: ModelResponse) -> bool:
    """Return True if any TextPart or ThinkingPart in `response` contains `OUTGOING_FORBIDDEN_SIGNATURE`."""
    for part in response.parts:
        if isinstance(
            part, (TextPart, ThinkingPart)
        ) and OUTGOING_FORBIDDEN_SIGNATURE in (part.content or ""):
            return True
    return False


@hooks.on.model_request
async def block_incoming_derp_eicar(
    ctx: RunContext[Any],
    *,
    request_context: ModelRequestContext,
    handler: Any,
) -> ModelResponse:
    """Intercept model requests and refuse to send to the LLM if `--DERPDERPEICAR--` is present."""
    last_response_idx = -1
    for idx, msg in enumerate(request_context.messages):
        if isinstance(msg, ModelResponse):
            last_response_idx = idx

    for idx, msg in enumerate(request_context.messages):
        if not isinstance(msg, ModelRequest):
            continue
        is_current_turn = idx > last_response_idx
        for part in msg.parts:
            if not isinstance(part, UserPromptPart):
                continue
            text = _extract_user_prompt_text(part)
            if INCOMING_FORBIDDEN_SIGNATURE in text:
                if is_current_turn:
                    return ModelResponse(
                        parts=[TextPart(content=INCOMING_REFUSAL_MESSAGE)]
                    )
                _redact_user_prompt_part(part, INCOMING_FORBIDDEN_SIGNATURE)

    return await handler(request_context)


@hooks.on.after_model_request
async def block_outgoing_derp_eicar_response(
    ctx: RunContext[Any],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    """Replace outgoing model responses containing `--DERPDERPEICAROUTGOING--` with a refusal message."""
    if _response_contains_outgoing_forbidden(response):
        return replace(response, parts=[TextPart(content=OUTGOING_REFUSAL_MESSAGE)])
    return response


@hooks.on.after_tool_execute
async def block_outgoing_derp_eicar_tool_result(
    ctx: RunContext[Any],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    result: Any,
) -> Any:
    """Prevent tool execution results containing `--DERPDERPEICAROUTGOING--` from being returned."""
    if OUTGOING_FORBIDDEN_SIGNATURE in str(result):
        return OUTGOING_REFUSAL_MESSAGE
    return result


@hooks.on.after_output_process
async def block_outgoing_derp_eicar_output(
    ctx: RunContext[Any],
    *,
    output_context: OutputContext,
    output: Any,
) -> Any:
    """Prevent `--DERPDERPEICAROUTGOING--` from being returned in the final processed output."""
    if isinstance(output, str) and OUTGOING_FORBIDDEN_SIGNATURE in output:
        return OUTGOING_REFUSAL_MESSAGE
    return output


@hooks.on.run_event_stream
async def filter_outgoing_derp_eicar_stream(
    ctx: RunContext[Any],
    *,
    stream: AsyncIterable[AgentStreamEvent],
) -> AsyncIterable[AgentStreamEvent]:
    """Hold back partial prefix matches in streaming deltas so `--DERPDERPEICAROUTGOING--` never leaks."""
    pending_by_index: dict[int, str] = {}
    blocked = False

    async for event in stream:
        if blocked:
            continue

        if isinstance(event, PartStartEvent) and isinstance(
            event.part, (TextPart, ThinkingPart)
        ):
            combined = event.part.content or ""
            if OUTGOING_FORBIDDEN_SIGNATURE in combined:
                blocked = True
                yield replace(event, part=TextPart(content=OUTGOING_REFUSAL_MESSAGE))
                continue
            hold = _longest_prefix_suffix_len(combined, OUTGOING_FORBIDDEN_SIGNATURE)
            safe = combined[: len(combined) - hold] if hold else combined
            pending_by_index[event.index] = (
                combined[len(combined) - hold :] if hold else ""
            )
            yield replace(event, part=replace(event.part, content=safe))
            continue

        if isinstance(event, PartDeltaEvent) and isinstance(
            event.delta, (TextPartDelta, ThinkingPartDelta)
        ):
            delta_text = event.delta.content_delta or ""
            combined = pending_by_index.get(event.index, "") + delta_text
            if OUTGOING_FORBIDDEN_SIGNATURE in combined:
                blocked = True
                pending_by_index.pop(event.index, None)
                yield PartDeltaEvent(
                    index=event.index,
                    delta=TextPartDelta(
                        content_delta=f"\n\n{OUTGOING_REFUSAL_MESSAGE}"
                    ),
                )
                continue
            hold = _longest_prefix_suffix_len(combined, OUTGOING_FORBIDDEN_SIGNATURE)
            safe = combined[: len(combined) - hold] if hold else combined
            pending_by_index[event.index] = (
                combined[len(combined) - hold :] if hold else ""
            )
            if safe:
                yield replace(event, delta=replace(event.delta, content_delta=safe))
            continue

        if isinstance(event, PartEndEvent) and isinstance(
            event.part, (TextPart, ThinkingPart)
        ):
            rem = pending_by_index.pop(event.index, "")
            if rem:
                delta_cls = (
                    ThinkingPartDelta
                    if isinstance(event.part, ThinkingPart)
                    else TextPartDelta
                )
                yield PartDeltaEvent(
                    index=event.index, delta=delta_cls(content_delta=rem)
                )
            if OUTGOING_FORBIDDEN_SIGNATURE in (event.part.content or ""):
                blocked = True
                yield replace(event, part=TextPart(content=OUTGOING_REFUSAL_MESSAGE))
                continue

        yield event
